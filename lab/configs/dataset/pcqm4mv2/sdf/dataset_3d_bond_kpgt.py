_base_ = [
    '../dataset_2d_kpgt.py',
]

data = dict(
    args=dict(
        dataset="PCQM4Mv23DDataset",
        dataset_args=dict(
            conformers_memmap='/data/pcqm4mv2_datahub/conformers/conformerpool_10conf_100iter_energymin.np',
            conformer_pool_size=1,
            root_dir='/data/pcqm4mv2_datahub/datasets/3d_bond_sdf'
        ),
        root_dir='/data/pcqm4mv2_datahub/datasets/3d_bond_sdf',
        collate_fn='default_collate_fn_with_kpgt',
        transform_configs=[
            dict(type='ComenetEdgeFeatures', args=dict(edge_index_key='edge_index', concatenate_with_edge_attr=True))
        ]
    )
)
