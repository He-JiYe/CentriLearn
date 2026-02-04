"""
Q-value head for Q-learning.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from centrilearn.utils.registry import HEADS

from .mlp_head import MLPHead


@HEADS.register_module()
class QHead(nn.Module):
    """Q-value prediction head.

    Args:
        in_channels: Input feature dimension.
        hidden_layers: List of hidden layer dimensions. Default: [in_channels, in_channels, 1].
        activation: Activation function.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_layers: list = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_channels, in_channels, 1]

        self.mlp = MLPHead(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
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
        assert info.get("node_embed") is not None, "node_embed is required"
        assert info.get("batch") is not None, "batch is required"
        assert info.get("graph_embed") is not None, "graph_embed is required"

        node_embed, batch, graph_embed = (
            info.get("node_embed"),
            info.get("batch"),
            info.get("graph_embed"),
        )

        if graph_embed is not None:
            node_embed = torch.cat([node_embed, graph_embed[batch]], dim=1)

        info["q_values"] = self.mlp(node_embed)
        return info
