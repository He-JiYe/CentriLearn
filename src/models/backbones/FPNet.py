"""
Feature Pyramid network backbone for graph reinforcement learning.
"""
import torch.nn as nn
import torch.nn.functional as F
from ..nn.GraphSAGE import GraphSAGE
from ..utils.registry import BACKBONES


@BACKBONES.register_module()
class FPNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)