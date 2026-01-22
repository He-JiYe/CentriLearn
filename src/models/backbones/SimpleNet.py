"""
Simple network backbone for graph reinforcement learning.
"""
import torch.nn as nn
import torch.nn.functional as F
from ..nn.GraphSAGE import GraphSAGE
from ..utils.registry import BACKBONES


@BACKBONES.register_module()
class SimpleNet(nn.Module):
    """Simple network backbone for graph reinforcement learning.

    This backbone uses a single GraphSAGE encoder for graph feature extraction.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_layers: Number of GraphSAGE layers.
        output_dim: Output feature dimension (Default: hidden_dim).
        aggr: GraphSAGE aggregation method ('mean', 'max', 'sum').
        graph_aggr: Graph pooling method ('add', 'mean', 'max').
        norm: Normalization type ('layer' or None).
        dropout: Dropout probability.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 output_dim: int = None,
                 aggr: str = 'mean',
                 graph_aggr: str = 'add',
                 norm: str = 'layer',
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else hidden_dim

        # Embedding layer
        self.fc = nn.Linear(input_dim, hidden_dim)

        # GraphSAGE encoder
        self.conv = GraphSAGE(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            output_channels=output_dim,
            aggr=aggr,
            graph_aggr=graph_aggr,
            norm=norm,
            dropout=dropout,
            **kwargs
        )

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            **kwargs: Other optional keys

        Returns:
            Node features [num_nodes, output_dim]
            Graph embedding [num_graphs, output_dim]
        """
        # Node to embedding
        x = F.relu(self.fc(x))

        # Graph encoding
        node_embed, graph_embed = self.conv(x, edge_index, batch)

        return node_embed, graph_embed

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]

        Returns:
            Dictionary containing:
                - node_embed: Node features [num_nodes, output_dim]
                - graph_embed: Graph embedding [num_graphs, output_dim]
        """
        x = info['x']
        edge_index = info['edge_index']
        batch = info['batch']

        # Node to embedding
        x = F.relu(self.fc(x))

        # Graph encoding
        node_embed, graph_embed = self.conv(x, edge_index, batch)

        # Update info
        info['node_embed'] = node_embed
        info['graph_embed'] = graph_embed

        return info

    @property
    def out_channels(self):
        """Output channels dimension."""
        return self.output_dim if self.output_dim else self.hidden_dim
