# - cached

_base_ = [
    './dataset_2d_linegraph.py',
]

data = dict(
    args=dict(
        transform_configs=[
            dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['fingerprint', 'molecule_descriptor', 'y']))
        ],
        dataset_args=dict(
            fingerprint=True,
            descriptor=True,
            fingerprint_memmap='/data/pcqm4mv2_datahub/kpgt_fingerprint_and_descriptor/fingerprint.np',
            descriptor_memmap='/data/pcqm4mv2_datahub/kpgt_fingerprint_and_descriptor/fingerprint.np'
        )
    )
)


# - noncached

# _base_ = [
#     './dataset_2d_kpgt.py',
# ]
#
# data = dict(
#     args=dict(
#         transform_configs=[
#             dict(type='LineGraphTransform', args=dict(bring_in_adjacent_nodes=True, keep_as_is=['fingerprint', 'molecule_descriptor', 'y']))
#         ]
#     )
# )
