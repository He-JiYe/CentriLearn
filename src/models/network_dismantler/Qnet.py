"""
Q-network model for network dismantling problem.
"""
import torch.nn as nn
import torch.nn.functional as F
from src.utils import NETWORK_DISMANTLER, build_backbone, build_head
from typing import Dict, Any

@NETWORK_DISMANTLER.register_module()
class Qnet(nn.Module):
    """Q-network for network dismantling tasks.

    This model combines a graph backbone with a Q-value prediction head.

    Args:
        backbone_cfg: Backbone configuration dictionary.
        q_head_cfg: Q-value head configuration.
    """

    def __init__(self,
                 backbone_cfg: dict,
                 q_head_cfg: dict = None):
        super().__init__()

        # Build backbone
        self.backbone = build_backbone(backbone_cfg)

        # Get backbone output dimension
        if hasattr(self.backbone, 'output_dim'):
            self.output_dim = self.backbone.output_dim
        else:
            raise ValueError("Backbone must have 'output_dim' property")

        # Build Q-value head
        if q_head_cfg is None:
            q_head_cfg = {
                'type': 'QHead',
                'input_dim': self.output_dim,
                'hidden_layers': [1],
            }

        self.q_head = build_head(q_head_cfg)

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Dictionary containing:
                q_values: Q-values for each node [num_nodes, 1]
                node_embed: Node embeddings [num_nodes, output_dim]
                graph_embed: Graph embeddings [batch_size, output_dim]
        """
        info = self.backbone(info)
        info = self.q_head(info)
        return info