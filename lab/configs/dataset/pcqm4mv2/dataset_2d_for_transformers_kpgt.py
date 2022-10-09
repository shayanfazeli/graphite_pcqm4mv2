_base_ = [
    './dataset_2d_for_transformers_base.py',
]

data = dict(
    args=dict(
        kpgt=True,
        root_dir='/data/pcqm4mv2_kpgt/',
    )
)
