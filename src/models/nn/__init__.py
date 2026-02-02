"""
Custom Neural Network.
"""
from .GraphSAGE import GraphSAGE
from .GIN import GIN
from .GAT import GAT

__all__ = [
    'GraphSAGE',
    'GIN',
    'GAT']
