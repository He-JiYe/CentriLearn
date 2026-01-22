"""
Builder functions for models.
"""
import torch.nn as nn
from typing import Union, List, Dict
from .registry import BACKBONES, HEADS, NETWORK_DISMANTLER


def build_backbone(cfg: Union[Dict, List], default_args: Dict = None):
    """Build backbone from config."""
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, BACKBONES, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, BACKBONES, default_args)

def build_head(cfg: Union[Dict, List], default_args: Dict = None):
    """Build head from config."""
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, HEADS, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, HEADS, default_args)

def build_network_dismantler(cfg: Union[Dict, List], default_args: Dict = None):
    """Build network_dismantler from config."""
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, NETWORK_DISMANTLER, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, NETWORK_DISMANTLER, default_args)

def build_from_cfg(cfg: Dict, registry, default_args: Dict = None):
    """Build a module from config dict.

    Args:
        cfg: Config dict. It should contain the key 'type'.
        registry: The registry to search the type from.
        default_args: Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')

    args = cfg.copy()

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)


# Import for type hints
import inspect
