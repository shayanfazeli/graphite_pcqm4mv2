_base_ = [
    './dataset_3d_bond.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            fingerprint=True,
            descriptor=True
        ),
    )
)
