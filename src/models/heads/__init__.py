"""
Heads module for graph neural networks.
"""
from .mlp_head import MLPHead
from .q_head import QHead
from .v_head import VHead
from .policy_head import PolicyHead
from .component_value import ComponentValueHead
from .dueling_head import DuelingHead

__all__ = [
    'MLPHead',
    'QHead',
    'VHead',
    'PolicyHead',
    'ComponentValueHead',
    'DuelingHead'
]
