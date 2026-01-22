"""
Value head for critic/advantage estimation.
"""
import torch.nn as nn
from .mlp_head import MLPHead
from ..utils.registry import HEADS


@HEADS.register_module()
class VHead(nn.Module):
    """Value estimation head for critic.

    Args:
        in_channels: Input feature dimension.
        hidden_layers: List of hidden layer dimensions. Default: [in_channels, in_channels, 1].
        activation: Activation function.
        dropout: Dropout probability.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: list = None,
                 activation: str = 'leaky_relu',
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_channels, in_channels, 1]

        self.mlp = MLPHead(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout
        )

    def forward(self, graph_embed, batch=None, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            graph_embed: graph features [num_graphs, in_channels]
            batch: batch assignment [num_nodes] (Default: None)
            **kwargs: Other optional keys
            
        Returns:
            If batch is None, return value estimation for each graph.
            If batch is not None, return value estimation for each node.
        """
        if batch is None:
            return self.mlp(graph_embed)
        else:
            return self.mlp(graph_embed[batch])

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - node_embed: Node features [num_nodes, in_channels]
                - graph_embed: Graph embeddings [num_graphs, in_channels]
                - batch: Batch assignment [num_nodes]
                - Other optional keys

        Returns:
            Updated info dictionary with value
        """
        graph_embed = info.get('graph_embed')
        batch = info.get('batch')

        if batch is None:
            value = self.mlp(graph_embed)
        else:
            value = self.mlp(graph_embed[batch])

        info['value'] = value
        return info

    @property
    def out_channels(self):
        """Output channels dimension."""
        return self.mlp.out_channels
