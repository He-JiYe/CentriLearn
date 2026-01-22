"""
Deep network backbone for graph reinforcement learning.
"""
import torch.nn as nn
import torch.nn.functional as F
from ..nn.GraphSAGE import GraphSAGE
from ..utils.registry import BACKBONES


@BACKBONES.register_module()
class DeepNet(nn.Module):
    def __init__(self):
        pass