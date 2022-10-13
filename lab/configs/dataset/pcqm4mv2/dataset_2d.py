data = dict(
    type='Pyg2DPCQM4Mv2',
    args=dict(
        batch_size=512,
        root_dir='/data/pcqm4mv2_datahub/datasets/2d',
        dataset="PCQM4Mv2DatasetFull",
        transform_configs=[],
        dataloader_base_args=dict(
            pin_memory=True,
            persistent_workers=True,
            num_workers=18
        ),
        collate_fn='default_collate_fn'
    )
)

