_base_ = [
    '../../../../../dataset/pcqm4mv2/sdf/dataset_3d_bond_doubleview.py',
    '../../../../../model/coatgin/coatgin_base2_3d_se3ddm.py',
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
            dict(type='Position3DGaussianNoise', args=dict(scale=0.1)),
            dict(type='ConcatenateAtomPositionsToEdgeAttributes'),
            dict(type='PairwiseDistances', args=dict()),
        ]
    )
)

model = dict(
    args=dict(
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
