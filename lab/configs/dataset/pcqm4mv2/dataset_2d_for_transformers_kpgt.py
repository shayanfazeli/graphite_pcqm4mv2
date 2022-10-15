_base_ = [
    './dataset_2d_for_transformers_base.py',
]

data = dict(
    args=dict(
        dataset_args=dict(
            fingerprint=True,
            descriptor=True
        ),
        root_dir='/data/pcqm4mv2_datahub/datasets/2d_kpgt/',
    )
)
