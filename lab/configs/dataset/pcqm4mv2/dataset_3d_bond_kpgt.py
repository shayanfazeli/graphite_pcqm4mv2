_base_ = [
    './dataset_3d_bond.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            fingerprint=True,
            descriptor=True
        ),
        collate_fn='default_collate_fn_with_kpgt'
    )
)
