"""
Custom Neural Network.
"""

from .GAT import GAT
from .GIN import GIN
from .GraphSAGE import GraphSAGE

__all__ = ["GraphSAGE", "GIN", "GAT"]
