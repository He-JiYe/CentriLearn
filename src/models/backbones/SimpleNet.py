"""
Simple network backbone for graph reinforcement learning.
"""
import torch.nn as nn
import torch.nn.functional as F
from src.models.nn.GraphSAGE import GraphSAGE
from src.utils.registry import BACKBONES
from typing import Dict, Any

@BACKBONES.register_module()
class SimpleNet(nn.Module):
    """Simple network backbone for graph reinforcement learning.

    This backbone uses a single GraphSAGE encoder for graph feature extraction.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden feature dimension.
        num_layers: Number of GraphSAGE layers.
        output_dim: Output feature dimension (Default: hidden_channels).
        aggr: GraphSAGE aggregation method ('mean', 'max', 'sum').
        graph_aggr: Graph pooling method ('add', 'mean', 'max').
        norm: Normalization type ('layer' or None).
        dropout: Dropout probability.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 output_dim: int = None,
                 aggr: str = 'mean',
                 graph_aggr: str = 'add',
                 norm: str = 'layer',
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._output_dim = output_dim if output_dim else hidden_channels

        # Embedding layer
        self.fc = nn.Linear(in_channels, hidden_channels)

        # GraphSAGE encoder
        self.conv = GraphSAGE(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            output_dim=output_dim,
            aggr=aggr,
            graph_aggr=graph_aggr,
            norm=norm,
            dropout=dropout,
            **kwargs
        )

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            node_embed: Node embeddings [num_nodes, hidden_channels]
            graph_embed: Graph embeddings [num_graphs, hidden_channels]
        """
        assert info.get('x') is not None, "x are required"
        assert info.get('edge_index') is not None, "Edge indices are required"
        assert info.get('batch') is not None, "Batch assignment is required"

        x, edge_index, batch = info['x'], info['edge_index'], info['batch']

        x = F.relu(self.fc(x))
        info['node_embed'], info['graph_embed'] = self.conv(x, edge_index, batch)
        return info
    
    @property
    def output_dim(self):
        """Output channels dimension."""
        return self._output_dim
