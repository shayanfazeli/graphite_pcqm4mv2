from typing import Dict, Union, Any
import collections.abc
import json
import argparse
from graphite.contrib.mmcv import Config


def nested_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def read_config(args, convert_to_dict: bool = True) -> Union[Config, Dict[str, Any]]:
    config = read_config_filepath(
        filepath=args.config,
        convert_to_dict=convert_to_dict
    )
    config = apply_overrides(config, args.config_overrides)
    return config


def read_config_filepath(filepath, convert_to_dict: bool = True) -> Union[Config, Dict[str, Any]]:
    if convert_to_dict:
        return dict(Config.fromfile(filepath))
    else:
        return Config.fromfile(filepath)


def apply_overrides(config, overrides):
    if overrides is not None:
        config = nested_dict_update(config, json.loads(overrides))
    return config
