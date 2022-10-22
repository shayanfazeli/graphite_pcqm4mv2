import numpy as np
import torch
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf

from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from .modified_ogb import AtomEncoder, BondEncoder
from graphite.cortex.model.model.mlp import CustomMLPHead


class ScaleLayer(nn.Module):
    """
    DeepNet scaling: [https://arxiv.org/abs/2203.00555v1](https://arxiv.org/abs/2203.00555v1)

    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    scale_init: `float`, required
        The initial value for scaling layer
    """
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(width) + np.log(scale_init))

    def forward(self, x):
        return pt.exp(self.scale) * x


class ScaleDegreeLayer(nn.Module):
    """
    DeepNet scaling: [https://arxiv.org/abs/2203.00555v1](https://arxiv.org/abs/2203.00555v1)

    Different for different node degrees.

    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    scale_init: `float`, required
        The initial value for scaling layer
    """
    def __init__(self, width, scale_init, max_degree: int = 4):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(max_degree, width) + np.log(scale_init))

    def forward(self, x, d):
        return pt.exp(self.scale)[d] * x


class GatedLinearBlock(nn.Module):
    """
    GLU: [https://arxiv.org/abs/1612.08083v3](https://arxiv.org/abs/1612.08083v3)

    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    width_head: `int`, required
        The number of heads (`width` must be divisible by it)

    width_scale: `int`, required
        The bottleneck scaling
    """
    def __init__(self, width: int, width_head: int, width_scale: int):
        super().__init__()
        num_grp = width // width_head
        self.pre = nn.Sequential(nn.Conv1d(width, width, 1),
                                 nn.GroupNorm(num_grp, width, affine=False))
        self.gate = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.value = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.post = nn.Conv1d(width * width_scale, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        xx = self.pre(x.unsqueeze(-1))
        xx = nnf.relu(self.gate(xx)) * self.value(xx)
        xx = self.post(xx).squeeze(-1)
        return xx


class ConvMessageComenet(MessagePassing):
    """
    VoVNet: [https://arxiv.org/abs/1904.09730v1](https://arxiv.org/abs/1904.09730v1)

    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    width_head: `int`, required
        The number of heads (`width` must be divisible by it)

    width_scale: `int`, required
        The bottleneck scaling

    hop: `int`, required
        hop

    kernel: `int`, required
        kernel

    scale_init: `float`, optional (default=0.1)
        The initial value for scaling layer
    """
    def __init__(
            self,
            width: int,
            width_head: int,
            width_scale: int,
            hop: int,
            kernel: int,
            scale_init: float = 0.1,
            pos_features: int = None,
    ):
        super().__init__(aggr="add")
        self.width = width
        self.hop = hop
        self.bond_encoder = nn.ModuleList()
        self.mlp = nn.ModuleList()
        if pos_features is not None:
            self.pos_encoder = nn.ModuleList()
        self.pos_features = pos_features
        self.scale = nn.ModuleList()
        for _ in range(hop * kernel):
            self.bond_encoder.append(BondEncoder(emb_dim=width))
            if pos_features is not None:
                self.pos_encoder.append(
                    CustomMLPHead(
                        input_dim=pos_features,
                        output_dim=width,
                        input_norm='BatchNorm1d',
                        num_hidden_layers=2,
                        hidden_dim=width//width_head,
                        activation='ReLU',
                        norm='LayerNorm',
                        output_norm='LayerNorm',
                        dropout=0.2
                    )
                )
            self.mlp.append(GatedLinearBlock(width, width_head, width_scale))
            self.scale.append(ScaleDegreeLayer(width, scale_init))

    def forward(self, x, node_degree, edge_index, edge_attr):
        for layer in range(len(self.mlp)):
            if layer == 0:
                x_raw, x_out = x, 0
            elif layer % self.hop == 0:
                x_raw, x_out = x + x_out, 0
            if self.pos_features is None:
                ea = self.bond_encoder[layer](edge_attr[:, :3].long())
            else:
                pos_mask = torch.any(edge_attr[:, 3:], dim=1, keepdim=True)
                ea = self.bond_encoder[layer](edge_attr[:, :3].long()) + pos_mask * self.pos_encoder[layer](edge_attr[:, 3:])

            x_raw = self.propagate(edge_index, x=x_raw, edge_attr=ea, layer=layer)
            x_out = x_out + self.scale[layer](x_raw, node_degree)
        return x_out

    def message(self, x_j, edge_attr, layer):
        return self.mlp[layer](x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


@torch.jit.script
def rbf(x, mean, std):
    # x : b, n, n, 128
    # mean: 1, 128
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianBasisFunctionsLayer(nn.Module):
    def __init__(self, K: int= 128, atom_types: int = 220, gbf_edge_attr: bool = False):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(atom_types, 1, padding_idx=0)
        self.bias = nn.Embedding(atom_types, 1, padding_idx=0)
        self.gbf_edge_attr = gbf_edge_attr
        if self.gbf_edge_attr:
            self.mul_edge_attr = nn.Embedding(18, 1, padding_idx=0)
            self.bias_edge_attr = nn.Embedding(18, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x: `torch.Tensor`, required
            The concatenated edge attribute of dim: `N, 3+3+3 + 1 + 1`
        """
        # N is total number of edges in the `Batch`
        edge_attr, pos_i, pos_j, atom_type_ij = torch.split(x, [3, 3, 3, 2], dim=1)
        dist = torch.linalg.norm(pos_i - pos_j, dim=1, ord=2, keepdim=True)  # N, 1
        if self.gbf_edge_attr:
            mul = self.mul(atom_type_ij).sum(dim=-2) + self.mul_edge_attr(edge_attr).sum(dim=-2) # N, 1
            bias = self.bias(atom_type_ij).sum(dim=-2) + self.bias_edge_attr(edge_attr).sum(dim=-2) # N, 1
        else:
            mul = self.mul(atom_type_ij).sum(dim=-2)  # N, 1
            bias = self.bias(atom_type_ij).sum(dim=-2)  # N, 1
        scaled_dist = mul * dist + bias  # N,1
        scaled_dist = scaled_dist.expand(-1, self.K)  # N, K

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2

        return rbf(scaled_dist, mean, std).type_as(self.means.weight)  # N, K


class ConvMessage(MessagePassing):
    """
    VoVNet: [https://arxiv.org/abs/1904.09730v1](https://arxiv.org/abs/1904.09730v1)

    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    width_head: `int`, required
        The number of heads (`width` must be divisible by it)

    width_scale: `int`, required
        The bottleneck scaling

    hop: `int`, required
        hop

    kernel: `int`, required
        kernel

    scale_init: `float`, optional (default=0.1)
        The initial value for scaling layer

    pos_features: `int`, optional (default=None)
        If a positive number, this means that there is 3d information and
        this conv message has to process it. In this case, the processing is done
        by creating the edge 3d features (using GBF) and then concatenation and
        mlp-merging with the encoded bond features.

    gbf_kernels: `int`, optional(default=128)
        GBF parameter

    gbf_edge_attr: `bool`, optional (default=False)
        GBF parameter
    """
    def __init__(
            self,
            width: int,
            width_head: int,
            width_scale: int,
            hop: int,
            kernel: int,
            scale_init: float = 0.1,
            pos_features: int = None,
            gbf_kernels: int = 128,
            gbf_edge_attr: bool = False
    ):
        super().__init__(aggr="add")
        self.width = width
        self.hop = hop
        self.bond_encoder = nn.ModuleList()
        self.mlp = nn.ModuleList()
        if pos_features is not None:
            self.pos_encoder = nn.ModuleList()
        self.pos_features = pos_features
        self.scale = nn.ModuleList()
        for _ in range(hop * kernel):
            self.bond_encoder.append(BondEncoder(emb_dim=width))
            if pos_features is not None:
                self.pos_encoder.append(
                    torch.nn.Sequential(
                        GaussianBasisFunctionsLayer(
                            K=gbf_kernels,
                            gbf_edge_attr=gbf_edge_attr
                        ),
                        CustomMLPHead(
                            input_dim=gbf_kernels,
                            output_dim=width,
                            input_norm='none',
                            num_hidden_layers=0,
                            hidden_dim=width,
                            activation='ReLU',
                            norm='none',
                            output_norm='none',
                            dropout=0.
                        )
                    )
                )
                self.merge_bond_with_pos_mlp = CustomMLPHead(
                            input_dim=2*width,
                            output_dim=width,
                            input_norm='none',
                            num_hidden_layers=1,
                            hidden_dim=width,
                            activation='ReLU',
                            norm='BatchNorm1d',
                            output_norm='LayerNorm',
                            dropout=0.1
                        )
            self.mlp.append(GatedLinearBlock(width, width_head, width_scale))
            self.scale.append(ScaleDegreeLayer(width, scale_init))

    def forward(self, x, node_degree, edge_index, edge_attr):
        for layer in range(len(self.mlp)):
            if layer == 0:
                x_raw, x_out = x, 0
            elif layer % self.hop == 0:
                x_raw, x_out = x + x_out, 0
            if self.pos_features is None:
                ea = self.bond_encoder[layer](edge_attr[:, :3].long())
            else:
                ea = self.merge_bond_with_pos_mlp(
                    torch.cat((
                            self.bond_encoder[layer](edge_attr[:, :3].long()),
                            self.pos_encoder[layer](edge_attr)), dim=-1)
                )

            x_raw = self.propagate(edge_index, x=x_raw, edge_attr=ea, layer=layer)
            x_out = x_out + self.scale[layer](x_raw, node_degree)
        return x_out

    def message(self, x_j, edge_attr, layer):
        return self.mlp[layer](x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class ConvMessageForLineGraph(MessagePassing):
    """
    Parameters
    ----------
    width: `int`, required
        The input/output dimension

    width_head: `int`, required
        The number of heads (`width` must be divisible by it)

    width_scale: `int`, required
        The bottleneck scaling

    hop: `int`, required
        hop

    kernel: `int`, required
        kernel

    scale_init: `float`, optional (default=0.1)
        The initial value for scaling layer
    """
    def __init__(self, width, width_head, width_scale, hop, kernel, scale_init=0.1):
        super().__init__(aggr="add")
        self.width = width
        self.hop = hop
        self.mlp = nn.ModuleList()
        self.scale = nn.ModuleList()
        for _ in range(hop * kernel):
            self.mlp.append(GatedLinearBlock(width, width_head, width_scale))
            self.scale.append(ScaleDegreeLayer(width, scale_init))

    def forward(self, x, node_degree, edge_index, edge_attr = None):
        for layer in range(len(self.mlp)):
            if layer == 0:
                x_raw, x_out = x, 0
            elif layer % self.hop == 0:
                x_raw, x_out = x + x_out, 0

            x_raw = self.propagate(edge_index, x=x_raw, layer=layer)
            x_out = x_out + self.scale[layer](x_raw, node_degree)
        return x_out

    def message(self, x_j, layer):
        return self.mlp[layer](x_j)

    def update(self, aggr_out):
        return aggr_out


# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width

        self.mlp = GatedLinearBlock(width, width_head, width_scale)
        self.scale = ScaleLayer(width, scale_init)

    def forward(self, x, x_res, batch, batch_size):
        xx = x_res = scatter(x, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xx = self.scale(self.mlp(xx))[batch]
        return xx, x_res


# CosFormer: https://openreview.net/pdf?id=Bl8CQrx2Up4
class AttMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width
        self.width_head = width_head

        num_grp = width // width_head
        self.pre = nn.Sequential(nn.Conv1d(width, width, 1),
                                 nn.GroupNorm(num_grp, width, affine=False))
        self.msgq = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.msgk = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.msgv = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.post = nn.Conv1d(width * width_scale, width, 1)
        self.scale = ScaleLayer(width, scale_init)

    def forward(self, x: torch.Tensor, x_res: torch.Tensor, batch: torch.Tensor, batch_size: int):
        xv = self.pre(x.unsqueeze(-1))

        shape = [len(x), -1, self.width_head]
        xq = pt.exp(self.msgq(xv) / np.sqrt(self.width_head)).reshape(shape)
        xk = pt.exp(self.msgk(xv) / np.sqrt(self.width_head)).reshape(shape)
        xv = self.msgv(xv).reshape(shape)

        xv = pt.einsum('bnh,bnv->bnhv', xk, xv)
        xv = x_res = scatter(xv, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xk = scatter(xk, batch, dim=0, dim_size=batch_size, reduce='sum')[batch]
        xq = xq / pt.einsum('bnh,bnh->bn', xq, xk)[:, :, None]  # norm
        xv = pt.einsum('bnh,bnhv->bnv', xq, xv[batch]).reshape(len(x), -1, 1)

        xv = self.scale(self.post(xv).squeeze(-1))
        return xv, x_res


class LineGraphNodeRepresentation(nn.Module):
    def __init__(self, width, width_head, width_scale, pos_features: int = None):
        super().__init__()
        self.bond_encoder = BondEncoder(emb_dim=width, left_padding=1)
        self.atom_encoder = nn.Sequential(
            AtomEncoder(emb_dim=width),
            GatedLinearBlock(width, width_head, width_scale))

        self.pos_features = pos_features

        if pos_features is not None:
            self.pos_encoder = CustomMLPHead(
                input_dim=pos_features,
                input_norm='BatchNorm1d',
                norm='LayerNorm',
                activation='ReLU',
                hidden_dim=width // width_head,
                num_hidden_layers=2,
                dropout=0.2,
                output_dim=width,
                output_norm='LayerNorm'
            )

        self.condenser = nn.Sequential(
            nn.BatchNorm1d(2 * width),
            nn.ReLU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width)
        )

    def forward(self, x):
        if self.pos_features is not None:
            bond_feats, pos_feats, atom1_feats, atom2_feats = torch.split(x, [3, self.pos_features, 9, 9], 1)
        else:
            bond_feats, atom1_feats, atom2_feats = torch.split(x, [3, 9, 9], 1)
        atom_feats = self.atom_encoder(atom1_feats.long()) + self.atom_encoder(atom2_feats.long())
        bond_feats = self.bond_encoder(bond_feats.long())
        condensed_reps = self.condenser(torch.cat((atom_feats, bond_feats), dim=1))

        if self.pos_features is not None:
            pos_mask = torch.any(pos_feats, dim=1, keepdim=True)
            condensed_reps = condensed_reps + pos_mask * self.pos_encoder(pos_feats)

        return condensed_reps


class CoAtGIN(pt.nn.Module):
    def __init__(
            self,
            num_layers: int,
            model_dim: int,
            conv_hop: int,
            conv_kernel: int,
            use_virt: bool = True,
            use_att: bool = True,
            max_degree: int = 4,
            line_graph: bool = False,
            num_heads: int = 16,
            expansion: int = 1,
            pos_features: int = None
    ):
        """
        Parameters
        ----------
        num_layers: `int`, required
            The number of layers

        model_dim: `int`, required
            The core dimensions

        conv_hop: `int`, required
            hops

        conv_kernel: `int`, required
            kernel

        line_graph: `bool`, required
            If true, node embeddings must have been already done elsewhere. otherwise, the atom encoding
            of OGB will be used prior to the model application.
        """
        super().__init__()
        self.num_layers = num_layers
        self.max_degree = max_degree
        self.line_graph = line_graph
        if self.line_graph:
            self.atom_encoder = LineGraphNodeRepresentation(width=model_dim, width_head=num_heads, width_scale=expansion, pos_features=pos_features)
        else:
            self.atom_encoder = AtomEncoder(model_dim)
        self.conv = pt.nn.ModuleList()
        self.virt = pt.nn.ModuleList()
        self.att = pt.nn.ModuleList()
        self.mlp = pt.nn.ModuleList()
        for layer in range(num_layers):
            if self.line_graph:
                self.conv.append(ConvMessageForLineGraph(
                    width=model_dim,
                    width_head=num_heads,
                    width_scale=expansion,
                    hop=conv_hop,
                    kernel=conv_kernel))
            else:
                self.conv.append(ConvMessage(
                    width=model_dim,
                    width_head=num_heads,
                    width_scale=expansion,
                    hop=conv_hop,
                    kernel=conv_kernel,
                    pos_features=pos_features
                ))
            self.virt.append(VirtMessage(model_dim, num_heads, 2) if use_virt else None)
            self.att.append(AttMessage(model_dim, num_heads, 2) if use_att else None)
            self.mlp.append(GatedLinearBlock(model_dim, num_heads, 3))

    def forward(self, batched_data: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        batch_size = len(batched_data.ptr) - 1
        node_degree = degree(edge_index[1], len(x)).long() - 1
        node_degree.clamp_(0, self.max_degree - 1)
        h_in, h_virt, h_att = self.atom_encoder(x), 0, 0

        for layer in range(self.num_layers):
            h_out = h_in + self.conv[layer](h_in, node_degree, edge_index, edge_attr)
            if self.virt[layer] is not None:
                h_tmp, h_virt = self.virt[layer](h_in, h_virt, batch, batch_size)
                h_out, h_tmp = h_out + h_tmp, None
            if self.att[layer] is not None:
                h_tmp, h_att = self.att[layer](h_in, h_att, batch, batch_size)
                h_out, h_tmp = h_out + h_tmp, None
            h_out = h_in = self.mlp[layer](h_out)

        return h_out
