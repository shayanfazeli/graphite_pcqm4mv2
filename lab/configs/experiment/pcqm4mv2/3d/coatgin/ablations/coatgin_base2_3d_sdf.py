_base_ = [
    '../../../../../dataset/pcqm4mv2/sdf/dataset_3d_bond.py',
    '../../../../../model/coatgin/coatgin_base2.py',
]

# - critical hyperparameters
__batch_size = 256
__warmup_epochs = 20
__max_epochs = 120
__learning_rate = 1e-2
__weight_decay = 2e-2

data = dict(
    args=dict(
        batch_size=__batch_size,
        transform_configs=[
            dict(type='ConcatenateAtomPositionsToEdgeAttributes')
        ]
    ),

)


model = dict(
    type="Regressor",
    args=dict(
        model_config=dict(
            args=dict(
                node_encoder_config=dict(args=dict(
                    pos_features=128,
                    gbf_kernels=128,
                    gbf_edge_attr=True,
                    pos_inclusion_strategy='random_contribution'
                ))
            )
        ),
        loss_config=dict(
            type='L1Loss',
            args=dict()
        ),
    )
)

# - override usual optimization
special_optimization = dict(
    type='coatgin_optim_and_scheduler',
    args=dict(
        lr=__learning_rate,
        wd=__weight_decay,
        warmups=__warmup_epochs
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
        validation_interval=1,
        metric_monitor=dict(
            mode='valid',
            metric='mae',
            direction='min'
        ),
        mixed_precision=False,
        mixed_precision_backend='none',
    )
)
