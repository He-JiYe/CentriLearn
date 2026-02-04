"""
dueling head for Q-learning.
"""
import torch
import torch.nn as nn
from .mlp_head import MLPHead
from .component_value import ComponentValueHead
from centrilearn.utils.registry import HEADS
from typing import Dict, Any

@HEADS.register_module()
class DuelingHead(nn.Module):
    """Dueling prediction head.

    Args:
        in_channels: Input feature dimension.
        hidden_layers: List of hidden layer dimensions. Default: [in_channels, in_channels, 1].
        activation: Activation function.
        dropout: Dropout probability.
        use_component: Whether to use component-level aggregation.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: list = None,
                 activation: str = 'leaky_relu',
                 dropout: float = 0.0,
                 use_component: bool = False):
        super().__init__()

        self.use_component = use_component

        if hidden_layers is None:
            hidden_layers = [in_channels, in_channels, 1]

        self.q_head = MLPHead(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout
        )

        if use_component:
            self.v_head = ComponentValueHead(
                in_channels=in_channels,
                hidden_layers=hidden_layers,
                activation=activation,
                dropout=dropout
            )
        else:
            self.v_head = MLPHead(
                in_channels=in_channels,
                hidden_layers=hidden_layers,
                activation=activation,
                dropout=dropout
            )

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).
        Args:
            node_embed: Node features [num_nodes, in_channels]
            batch: Batch assignment [num_nodes]
            graph_embed: Graph embeddings [batch_size, in_channels]

        Returns:
            Q-values [num_nodes, 1]
        """
        assert info.get('node_embed') is not None, "node_embed is required"
        assert info.get('batch') is not None, "batch is required"
        if self.use_component:
            assert info.get('graph_embed') is not None, "graph_embed is required if use_component is True"

        node_embed, batch, graph_embed = info.get('node_embed'), info.get('batch'), info.get('graph_embed')

        if not self.use_component:
            node_embed = torch.cat([node_embed, graph_embed[batch]], dim=1)

        advantage = self.q_head(node_embed)
        if self.use_component:
            value = self.v_head(info)
        else:
            value = self.v_head(node_embed)
        info['q_values'] = value + advantage - advantage.mean(dim=1, keepdim=True)
        return info