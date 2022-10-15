_base_ = [
    './dataset_2d.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            conformers_memmap='/data/pcqm4mv2_datahub/conformers/conformerpool_10conf_100iter_energymin.np',
            conformer_pool_size=1,
        ),
        transform_configs=[
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True))
        ]
    )
)
