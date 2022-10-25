"""
Exp10
"""

_base_ = [
    './config.py',
]


data = dict(
    args=dict(
        split_dict_filepath='/data/pcqm4mv2_datahub/splits/jp_split_fold_1.pt'
    )
)
