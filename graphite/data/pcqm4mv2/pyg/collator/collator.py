import torch
import torch.nn
from torch_geometric.loader.dataloader import Collater

from graphite.data.pcqm4mv2.pyg.transforms import LineGraphTransform
from graphite.data.utilities.sequence_collate.utilities import pad_sequence, pad_sequence_2d

default_collate_fn = Collater([], [
    'node2node_shortest_path_length_type',
    'node2node_connection_type',
    'shortest_path_feature_trajectory',
    'molecule_fingerprint',
    'molecule_descriptor',
    # 'positions_3d',
    'comenet_features1',
    'comenet_features2',
    'pairwise_distances'
])  # ([dataset_transformed[i] for i in [1,2,3]])


def collate_fn(batch):
    """collation"""
    no_edge_val = 8
    graphs = default_collate_fn(batch)
    node_counts = torch.LongTensor([graphs[i].num_nodes for i in range(len(batch))])
    max_node_count = node_counts.max().item()

    node_features = pad_sequence([g.x for g in batch], pad_dim=0, pad_value=0, stack_dim=0, max_len=max_node_count)[
        0].long()
    edge_type = pad_sequence([g.edge_attr for g in batch], pad_dim=0, pad_value=-1, stack_dim=0, max_len=None)[
        0].long()

    output = dict(
        node_features=node_features,
        edge_type=edge_type,
        graphs=graphs,
        node_counts=node_counts
    )

    if 'y' in graphs:
        output['y'] = graphs.y

    if 'node_type' in batch[0]:
        output['node_type'] = pad_sequence([g['node_type'] for g in batch], pad_dim=0, pad_value=0, stack_dim=0, max_len=max_node_count)[0].long()

    if 'node2node_shortest_path_length_type' in batch[0]:
        output['node2node_shortest_path_length_type'] = pad_sequence_2d(
            [g['node2node_shortest_path_length_type'] for g in batch],
            pad_dims=(0, 1),
            max_len=max_node_count,
            pad_value=no_edge_val,
            stack_dim=0)[0].long()
    if 'node2node_connection_type' in batch[0]:
        output['node2node_connection_type'] = pad_sequence_2d(
            [g['node2node_connection_type'] for g in batch],
            pad_dims=(0, 1),
            max_len=max_node_count,
            pad_value=30,
            stack_dim=0)[0].long()

    if 'shortest_path_feature_trajectory' in batch[0]:
        pad_value = -1
        shortest_path_feature_trajectory = []

        max_shortest_path_length = max([g['shortest_path_feature_trajectory'].shape[-2] for g in batch])

        for g in batch:
            tmp = g['shortest_path_feature_trajectory'][:-1, :, :, :]
            s = list(tmp.size())
            s[0] = max_node_count - s[0]
            tmp = torch.cat((tmp, torch.ones(*s) * pad_value), dim=0)
            s = list(tmp.size())
            s[1] = max_node_count - s[1]
            tmp = torch.cat((tmp, torch.ones(*s) * pad_value), dim=1)
            s = list(tmp.size())
            s[2] = max_shortest_path_length - s[2]
            tmp = torch.cat((tmp, torch.ones(*s) * pad_value), dim=2)
            shortest_path_feature_trajectory.append(tmp)

        output['shortest_path_feature_trajectory'] = torch.stack(shortest_path_feature_trajectory).long()

    if 'fingerprint' in batch[0]:
        output['molecule_fingerprint'] = torch.stack([g['fingerprint'] for g in batch], dim=0).float()

    if 'molecule_descriptor' in batch[0]:
        output['molecule_descriptor'] = torch.stack([g['molecule_descriptor'] for g in batch], dim=0)[:, 1:].float()
        output['molecule_descriptor'] = torch.nan_to_num(output['molecule_descriptor'], nan=0)

    # - 3d features
    for k in ['positions_3d', 'comenet_features1', 'comenet_features2']:
        if k in batch[0]:
            output[k] = pad_sequence(
                [g[k] for g in batch],
                stack_dim=0,
                pad_dim=0,
                pad_value=0,
                max_len=None  # infer
            )[0]

    if 'node_degree_centrality' in batch[0]:
        output['node_degree_centrality'] = pad_sequence([g['node_type'] for g in batch], pad_dim=0, pad_value=-1, stack_dim=0, max_len=max_node_count)[
            0].long()

    if 'pairwise_distances' in batch[0]:
        output['pairwise_distances'] = pad_sequence_2d(
            [g['pairwise_distances'] for g in batch], pad_value=-1, stack_dim=0, max_len=max_node_count
        )

    for k in output:
        if isinstance(k, torch.Tensor):
            assert not torch.any(torch.isnan(output[k])).item()

    return output


def default_collate_fn_with_kpgt(batch):
    fingerprint = torch.stack([g.fingerprint for g in batch])
    molecule_descriptor = torch.stack([g.molecule_descriptor for g in batch])

    if 'pairwise_distances' in batch[0]:
        pairwise_distances = pad_sequence_2d(
            [g['pairwise_distances'] for g in batch],
            pad_value=-1,
            stack_dim=0,
            max_len=max([g.num_nodes for g in batch])
        )
        for i in range(len(batch)):
            del batch[i].pairwise_distances

    g = default_collate_fn(batch)
    del g.fingerprint
    del g.molecule_descriptor
    del g.pairwise_distances

    g.molecule_fingerprint = fingerprint.float()
    g.molecule_descriptor = torch.nan_to_num(molecule_descriptor[:, 1:], nan=0).float()
    if 'pairwise_distances' in batch[0]:
        g.pairwise_distances = pairwise_distances
    g.node_counts = torch.tensor([g[i].num_nodes for i in range(len(g))])
    return g


def default_multiview_collate_fn(batch):
    return [default_collate_fn(list(e)) for e in list(zip(*batch))]


def default_multiview_collate_fn_with_kpgt(batch):
    return [default_collate_fn_with_kpgt(list(e)) for e in list(zip(*batch))]
