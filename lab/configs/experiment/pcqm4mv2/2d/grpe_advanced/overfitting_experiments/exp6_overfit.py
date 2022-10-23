"""
Exp6
"""

_base_ = [
    '../../../../../dataset/pcqm4mv2/dataset_2d_for_transformers_kpgt.py',
    '../../../../../model/grpe_advanced/grpe_large.py',
]

# - additional setup
__number_of_processes = 4  # this is not directly used, the caller has to make sure it is compatible.
__number_of_training_items = 3378606

# - critical hyperparameters
__batch_size = 256
__warmup_epochs = 3
__max_epochs = 400
__shortest_path_length_type_upperbound = 5  # for the shortest-path-type (discrete) to be embedded
__shortest_path_length_upperbound = 5  # for graphormer-like path embedding

__attention_biases = [
                    'edge',
                    'shortest_path_length',
                    'shortest_path'
                ]
__path_encoding_code_dim = 4
__encode_node_degree_centrality = True
__overfit_on_train_subset=10000

data = dict(
    args=dict(
        batch_size=__batch_size,
        overfit_on_train_subset=__overfit_on_train_subset,
        transform_configs=[
            dict(
                type='EncodeNode2NodeShortestPathFeatureTrajectory',
                args=dict(
                    max_length_considered=__shortest_path_length_type_upperbound, feature_position_offset=4
                )
            ),
            dict(type='EncodeNodeType'),
            dict(type='EncodeNodeDegreeCentrality'),
            dict(type='AddTaskNode'),
            dict(type='EncodeEdgeType'),
            dict(type='EncodeNode2NodeConnectionType'),
            dict(type='EncodeNode2NodeShortestPathLengthType'),
            dict(type='EncodeNode2NodeShortestPathLengthType', args=dict(max_length_considered=__shortest_path_length_upperbound))
        ],
    )
)

model = dict(
    type="RegressorWithKPGTRegularization",
    args=dict(
        model_config=dict(
            args=dict(
                shortest_path_length_upperbound=__shortest_path_length_type_upperbound,
                attention_biases=__attention_biases,
                path_encoding_length_upperbound=__shortest_path_length_upperbound,
                path_encoding_code_dim=__path_encoding_code_dim,
                encode_node_degree_centrality=__encode_node_degree_centrality
            )
        ),
        loss_config=dict(
            type='L1Loss',
            args=dict()
        ),
        kpgt_loss_config=dict(
            fingerprint=dict(
                factor=5e-2,
                type='BCEWithLogitsLoss',
                args=dict()
            ),
            descriptor=dict(
                factor=1e-1,
                type='L1Loss',
                args=dict()
            )
        )
    )
)

optimizer = dict(
    type='AdamW',
    args=dict(
        lr=2e-4,
        weight_decay=0
    ),
)

scheduler = dict(
    type='PolynomialDecayLR',
    args=dict(
        warmup_updates=__warmup_epochs * ((__number_of_training_items // __batch_size) // __number_of_processes),
        tot_updates=__max_epochs * ((__number_of_training_items // __batch_size) // __number_of_processes),
        lr=2e-4,
        end_lr=1e-9,
        power=1.0
    ),
    interval='step'
)

metrics = dict(
    train=[
        dict(
            name='loss_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss')
        ),
        dict(
            name='loss_kpgt_fp_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt_fp')
        ),
        dict(
            name='loss_kpgt_desc_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt_desc')
        ),
        dict(
            name='loss_kpgt_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt')
        ),
        dict(
            name='mae',
            type='MeanAbsoluteError',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(preds='preds', target='y')
        ),
    ],
    valid=[
        dict(
            name='loss_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss')
        ),
        dict(
            name='loss_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss')
        ),
        dict(
            name='loss_kpgt_fp_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt_fp')
        ),
        dict(
            name='loss_kpgt_desc_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt_desc')
        ),
        dict(
            name='loss_kpgt_mean',
            type='MeanMetric',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(value='loss_kpgt')
        ),
        dict(
            name='mae',
            type='MeanAbsoluteError',
            init_args=dict(dist_sync_on_step=False),
            arg_mapping=dict(preds='preds', target='y')
        ),
    ]
)

trainer = dict(
    type='Trainer',
    args=dict(
        modes=['train', 'valid'],
        max_epochs=__max_epochs,
        validation_interval=5,
        metric_monitor=dict(
            mode='valid',
            metric='mae',
            direction='min'
        ),
        mixed_precision=True,
        mixed_precision_backend='amp',
        # limit_batch_count=100
    )
)
