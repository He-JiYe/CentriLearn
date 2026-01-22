"""
TODO: Qnet model for network dismantling problem.
"""
import torch.nn as nn
from ..utils.registry import NETWORK_DISMANTLER
from ..utils.builder import build_backbone

@NETWORK_DISMANTLER.register_module()
class Qnet(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
    
    def forward(self, x):
        return self.backbone(x)