_base_ = [
    './2d_linegraph.py',
]


model = dict(
    type="RegressorWithKPGTRegularization",
    args=dict(
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

