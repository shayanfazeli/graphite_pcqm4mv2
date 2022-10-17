_base_ = [
    '../../../../dataset/pcqm4mv2/dataset_3d_bond_kpgt_linegraph.py',
    '../../../../model/coatgin/coatgin_base.py',
]

# - critical hyperparameters
__batch_size = 256
__warmup_epochs = 20
__max_epochs = 120
__learning_rate = 3e-3
__weight_decay = 2e-2

data = dict(
    args=dict(
        batch_size=__batch_size
    )
)


model = dict(
    type="RegressorWithKPGTRegularization",
    args=dict(
        model_config=dict(
            args=dict(
                node_encoder_config=dict(
                    args=dict(
                        pos_features=18,
                        line_graph=True
                    )
                )
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
