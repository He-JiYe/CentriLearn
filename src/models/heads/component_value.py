"""
Component value head for hierarchical value estimation.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_max, scatter_sum
from .mlp_head import MLPHead
from src.utils.registry import HEADS
from typing import Dict, Any

@HEADS.register_module()
class ComponentValueHead(nn.Module):
    """Component value head for hierarchical value estimation.

    Aggregates node values by component, then aggregates component values to graph level.

    Args:
        in_channels: Input node feature dimension.
        hidden_layers: List of hidden layer dimensions for component MLP.
        num_critics: Number of critics for conservative value estimation.
        activation: Activation function.
        dropout: Dropout probability.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: list = None,
                 activation: str = 'leaky_relu',
                 dropout: float = 0.0):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_channels, 1]

        self.mlp = MLPHead(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout
        )
        

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).

        Args:
            node_embed: Node embeddings [num_nodes, in_channels]
            batch: Batch assignment [num_nodes]
            component: Component assignment [num_nodes]

        Returns:
            Graph values [batch_size, 1]
        """
        assert info.get('node_embed') is not None, "node_embed is required"
        assert info.get('batch') is not None, "batch is required"
        assert info.get('component') is not None, "component is required"
        
        node_embed = info['node_embed']
        batch = info['batch']
        component = info['component']

        # Assign unique component indices
        num_component_per_graph = scatter_max(component, batch)[0] + 1
        offsets = torch.zeros_like(num_component_per_graph)
        offsets[1:] = torch.cumsum(num_component_per_graph[:-1], dim=0)
        component = component + offsets[batch]

        # component-level aggregation
        component_embed = global_add_pool(node_embed, component)

        # component value prediction
        component_value = self.mlp(component_embed)

        # Aggregate components for each graph
        indices = torch.arange(len(num_component_per_graph), device=node_embed.device)
        info['v_values'] = scatter_sum(component_value, torch.repeat_interleave(indices, num_component_per_graph), dim=0)

        return info
