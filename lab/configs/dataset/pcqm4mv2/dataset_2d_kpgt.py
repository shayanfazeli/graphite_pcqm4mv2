_base_ = [
    './dataset_2d.py',
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
