"""
Heads module for graph neural networks.
"""
from .mlp_head import MLPHead
from .q_head import QHead
from .v_head import VHead
from .logit_head import LogitHead
from .branch_value import BranchValueHead

__all__ = [
    'MLPHead',
    'QHead',
    'VHead',
    'LogitHead',
    'BranchValueHead'
]
