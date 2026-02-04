"""
Backbone networks for graph neural networks.
"""
from .GraphSAGE import GraphSAGE
from .GIN import GIN
from .GAT import GAT
from .DeepNet import DeepNet
from .FPNet import FPNet

__all__ = [
    'GraphSAGE', 
    'GIN',
    'GAT'
    'DeepNet', 
    'FPNet',
]