_base_ = [
    './dataset_2d.py',
]

data = dict(
    args=dict(
        transform_configs=[
            dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['fingerprint', 'molecule_descriptor', 'y']))
        ]
    )
)
