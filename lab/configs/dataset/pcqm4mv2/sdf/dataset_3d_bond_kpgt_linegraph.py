# _base_ = [
#     './dataset_3d_bond_kpgt.py',
# ]
#
# data = dict(
#     args=dict(
#         root_dir='/data/pcqm4mv2_datahub/datasets/3d_bond_linegraph',
#         transform_configs=[
#             # dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True)),
#             # dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['fingerprint', 'molecule_descriptor', 'y']))
#         ]
#     )
# )


# # - non cached
_base_ = [
    './dataset_3d_bond_kpgt.py',
]

data = dict(
    args=dict(
        transform_configs=[
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True)),
            dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['fingerprint', 'molecule_descriptor', 'y']))
        ]
    )
)
