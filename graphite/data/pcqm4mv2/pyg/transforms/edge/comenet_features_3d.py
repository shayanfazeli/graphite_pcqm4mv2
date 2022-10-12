from typing import List
import math
from torch_geometric.data import Data
import torch
import torch.nn
from torch_scatter import scatter, scatter_min
from graphite.contrib.comenet.positions import *
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform
from graphite.utilities.logging import get_logger

logger = get_logger(__name__)


class ComenetEdgeFeatures(BasePygGraphitePCQM4MTransform):
    """
    The 3D edge features of ComENet `https://arxiv.org/pdf/2206.08515.pdf`.

    # todo: modify for customizeable application on different edge indices (possibly multiple)
    """
    def __init__(
            self,
            cutoff=8.0,
            num_radial=3,
            num_spherical=2,
            edge_index_key: str = 'edge_index',
            concatenate_with_edge_attr: bool = False
    ):
        """constructor"""
        super(ComenetEdgeFeatures, self).__init__()
        self.cutoff = cutoff
        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.edge_index_key = edge_index_key
        self.concatenate_with_edge_attr = concatenate_with_edge_attr

    def forward(self, g: Data) -> Data:
        """
        Parameters
        ----------
        g: `Data`, required


        Returns
        ----------
        """
        # - required material
        pos = g.positions_3d
        j, i = tuple([e.squeeze() for e in torch.split(g[self.edge_index_key], 1, 0)])
        num_nodes = g.num_nodes

        if i.size(0) == 0:

            logger.warning(f"comenet features: no edge was there in the atom with {pos.shape[0]} nodes. setting placeholder.")

            if not self.concatenate_with_edge_attr:
                g['comenet_features1'] = torch.zeros((0, 12))
                g['comenet_features2'] = torch.zeros((0, 6))
            else:
                g['edge_attr'] = torch.cat((g.edge_attr, torch.zeros((0, 12)), torch.zeros((0, 6))), dim=1)
            return g

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        try:
            n0 = j[argmin0]
        except:
            import pdb
            pdb.set_trace()
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------
        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        if not self.concatenate_with_edge_attr:
            g['comenet_features1'] = self.feature1(dist, theta, phi)
            g['comenet_features2'] = self.feature2(dist, tau)
        else:
            g['edge_attr'] = torch.cat((g.edge_attr, self.feature1(dist, theta, phi), self.feature2(dist, tau)), dim=1)

        return g
