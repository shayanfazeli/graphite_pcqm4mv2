_base_ = [
    './dataset_2d_kpgt.py',
]

data = dict(
    args=dict(
        dataset='MultiviewPCQM4Mv2Dataset',
        dataset_args=dict(
            conformers_memmap='/data/pcqm4mv2_datahub/conformers/conformerpool_10conf_100iter_energymin.np',
            conformer_pool_size=1,  # we want to perturb the same conformer
            num_views=2,
        ),
        transform_configs=[
            dict(type='Position3DGaussianNoise', args=dict(scale=0.1)),
            dict(type='PairwiseDistances', args=dict()),
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True))
        ],
        collate_fn='default_multiview_collate_fn_with_kpgt'
    )
)
