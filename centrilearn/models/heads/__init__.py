"""
Heads module for graph neural networks.
"""

from .component_value import ComponentValueHead
from .dueling_head import DuelingHead
from .mlp_head import MLPHead
from .policy_head import PolicyHead
from .q_head import QHead
from .v_head import VHead

__all__ = [
    "MLPHead",
    "QHead",
    "VHead",
    "PolicyHead",
    "ComponentValueHead",
    "DuelingHead",
]
