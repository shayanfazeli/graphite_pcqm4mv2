_base_ = [
    './dataset_2d.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            conformers_memmap='/data/conformers.np',
            num_conformers_to_return=2,
        ),
        transform_configs=[
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True))
        ]
    )
)
