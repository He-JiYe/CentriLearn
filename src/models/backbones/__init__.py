"""
Backbone networks for graph neural networks.
"""
from .SimpleNet import SimpleNet
from .DeepNet import DeepNet
from .FPNet import FPNet

__all__ = [
    'SimpleNet', 
    'DeepNet', 
    'FPNet',
]