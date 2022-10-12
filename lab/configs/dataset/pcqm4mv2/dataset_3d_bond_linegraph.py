_base_ = [
    './dataset_3d_bond.py',
]

data = dict(
    args=dict(
        transform_configs=[
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True)),
            dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['y']))
        ]
    )
)