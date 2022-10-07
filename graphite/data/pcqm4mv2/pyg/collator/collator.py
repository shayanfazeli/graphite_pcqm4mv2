import torch
import torch.nn
from torch_geometric.loader.dataloader import Collater
from graphite.data.utilities.sequence_collate.utilities import pad_sequence, pad_sequence_2d

default_collate_fn = Collater([], [
    'node2node_shortest_path_length_type',
    'node2node_connection_type',
    'shortest_path_feature_trajectory',
    'molecule_fingerprint',
    'molecule_descriptor',
    'positions_3d',
    'comenet_features1',
    'comenet_features2'
])  # ([dataset_transformed[i] for i in [1,2,3]])


def collate_fn(batch):
    """collation"""
    no_edge_val = 8
    graphs = default_collate_fn(batch)
    node_counts = torch.LongTensor([graphs[i].num_nodes for i in range(len(batch))])
    max_node_count = node_counts.max().item()

    node_type = pad_sequence([g['node_type'] for g in batch], pad_dim=0, pad_value=0, stack_dim=0, max_len=max_node_count)[0].long()
    node_features = pad_sequence([g.x for g in batch], pad_dim=0, pad_value=0, stack_dim=0, max_len=max_node_count)[
        0].long()
    edge_type = pad_sequence([g.edge_attr for g in batch], pad_dim=0, pad_value=259, stack_dim=0, max_len=None)[
        0].long()

    output = dict(
        node_type=node_type,
        node_features=node_features,
        edge_type=edge_type,
        graphs=graphs,
        node_counts=node_counts
    )

    if 'y' in graphs:
        output['y'] = graphs.y

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

    if 'molecule_fingerprint' in batch[0]:
        output['molecule_fingerprint'] = torch.stack([g['molecule_fingerprint'] for g in batch], dim=0)

    if 'molecule_descriptor' in batch[0]:
        output['molecule_descriptor'] = torch.stack([g['molecule_descriptor'] for g in batch], dim=0)

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

    for k in output:
        if isinstance(k, torch.Tensor):
            assert not torch.any(torch.isnan(output[k])).item()

    return output
