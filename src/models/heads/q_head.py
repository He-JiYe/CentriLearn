"""
Q-value head for Q-learning.
"""
import torch
import torch.nn as nn
from .mlp_head import MLPHead
from ..utils.registry import HEADS


@HEADS.register_module()
class QHead(nn.Module):
    """Q-value prediction head.

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

    def forward(self, node_embed, batch, graph_embed=None, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            node_embed: Node features [num_nodes, in_channels]
            batch: Batch assignment [num_nodes]
            graph_embed: Graph embeddings [batch_size, in_channels]
            **kwargs: Other optional keys
            
        Returns:
            Q-values [num_nodes, 1]
        """
        if graph_embed is not None:
            node_embed = torch.cat([node_embed, graph_embed[batch]], dim=1)

        return self.mlp(node_embed)

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - node_embed: Node features [num_nodes, in_channels]
                - graph_embed: Graph embeddings [batch_size, in_channels] (optional)
                - batch: Batch assignment [num_nodes]
                - Other optional keys

        Returns:
            Updated info dictionary with q_values
        """
        x = info['node_embed']
        batch = info.get('batch')

        if 'graph_embed' in info and info['graph_embed'] is not None:
            x = torch.cat([x, info['graph_embed'][batch]], dim=1)

        q_values = self.mlp(x)
        info['q_values'] = q_values
        return info

    @property
    def out_channels(self):
        """Output channels dimension."""
        return self.mlp.out_channels
