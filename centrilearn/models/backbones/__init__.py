"""
Backbone networks for graph neural networks.
"""

from .DeepNet import DeepNet
from .FPNet import FPNet
from .GAT import GAT
from .GIN import GIN
from .GraphSAGE import GraphSAGE

__all__ = [
    "GraphSAGE",
    "GIN",
    "GAT" "DeepNet",
    "FPNet",
]
