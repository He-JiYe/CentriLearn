"""
Policy head for policy/action probability.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from centrilearn.utils.registry import HEADS

from .mlp_head import MLPHead


@HEADS.register_module()
class PolicyHead(nn.Module):
    """Policy head for policy/action probability.

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
            Action logits [num_nodes, num_actions]
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

        info["logit"] = self.mlp(node_embed)
        return info

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - node_embed: Node features [num_nodes, in_channels]
                - graph_embed: Graph embeddings [batch_size, in_channels] (optional)
                - batch: Batch assignment [num_nodes]
                - Other optional keys

        Returns:
            Updated info dictionary with logit
        """
        x = info["node_embed"]
        batch = info.get("batch")

        if "graph_embed" in info and info["graph_embed"] is not None:
            x = torch.cat([x, info["graph_embed"][batch]], dim=1)

        logit = self.mlp(x)
        info["logit"] = logit
        return info
