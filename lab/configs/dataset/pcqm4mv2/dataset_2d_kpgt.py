_base_ = [
    './dataset_2d.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            fingerprint=True,
            descriptor=True
        ),
        root_dir='/data/pcqm4mv2_datahub/datasets/2d_kpgt',
        collate_fn='default_collate_fn_with_kpgt'
    )
)
