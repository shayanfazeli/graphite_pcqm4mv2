data = dict(
    type='Pyg2DPCQM4Mv2',
    args=dict(
        batch_size=256,
        root_dir='/home/shayan/from_source/GRPE/data',
        transform_configs=[
            # dict(type='EncodeNode2NodeShortestPathFeatureTrajectory'),
            dict(type='EncodeNodeType'),
            dict(type='AddTaskNode'),
            dict(type='EncodeEdgeType'),
            dict(type='EncodeNode2NodeConnectionType'),
            dict(type='EncodeNode2NodeShortestPathLengthType'),
        ],
        dataloader_base_args=dict(
            pin_memory=True,
            persistent_workers=True,
            num_workers=10
        )
    )
)

