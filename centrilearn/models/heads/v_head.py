"""
Value head for critic/advantage estimation.
"""

from typing import Any, Dict

from torch import Tensor, nn

from centrilearn.utils.registry import HEADS

from .mlp_head import MLPHead


@HEADS.register_module()
class VHead(nn.Module):
    """Value estimation head for critic.

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
            graph_embed: graph features [num_graphs, in_channels]
            batch: batch assignment [num_nodes] (Default: None)

        Returns:
            v_values estimation for each node in graph [num_nodes, 1].
        """
        assert info.get("graph_embed") is not None, "graph_embed is required"
        graph_embed, batch = info.get("graph_embed"), info.get("batch")
        info["v_values"] = self.mlp(graph_embed)

        return info
