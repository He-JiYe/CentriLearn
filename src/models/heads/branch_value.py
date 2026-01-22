"""
Branch value head for hierarchical value estimation.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_max, scatter_sum
from .mlp_head import MLPHead
from ..utils.registry import HEADS


@HEADS.register_module()
class BranchValueHead(nn.Module):
    """Branch value head for hierarchical value estimation.

    Aggregates node values by branch, then aggregates branch values to graph level.

    Args:
        in_channels: Input node feature dimension.
        hidden_layers: List of hidden layer dimensions for branch MLP.
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
        

    def forward(self, node_embed, batch, branch, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            node_embed: Node embeddings [num_nodes, in_channels]
            batch: Batch assignment [num_nodes]
            branch: Branch assignment [num_nodes]
            **kwargs: Other optional keys
            
        Returns:
            Graph values [batch_size, 1]
        """
        # Assign unique branch indices
        num_branch_per_graph = scatter_max(branch, batch)[0] + 1
        offsets = torch.zeros_like(num_branch_per_graph)
        offsets[1:] = torch.cumsum(num_branch_per_graph[:-1], dim=0)
        branch = branch + offsets[batch]

        # Branch-level aggregation
        branch_embed = global_add_pool(node_embed, branch)

        # Branch value prediction
        branch_value = self.mlp(branch_embed)

        # Aggregate branches for each graph
        indices = torch.arange(len(num_branch_per_graph), device=node_embed.device)
        graph_value = scatter_sum(branch_value, torch.repeat_interleave(indices, num_branch_per_graph), dim=0)

        return graph_value

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - node_embed: Node embeddings [num_nodes, in_channels]
                - graph_embed: Graph embeddings [batch_size, in_channels]
                - batch: Batch assignment [num_nodes]
                - branch: Branch assignment [num_nodes]
                - Other optional keys

        Returns:
            Updated info dictionary with graph_value
        """
        node_embed = info['node_embed']
        batch = info['batch']
        branch = info['branch']

        # Assign unique branch indices
        num_branch_per_graph = scatter_max(branch, batch)[0] + 1
        offsets = torch.zeros_like(num_branch_per_graph)
        offsets[1:] = torch.cumsum(num_branch_per_graph[:-1], dim=0)
        branch = branch + offsets[batch]

        # Branch-level aggregation
        branch_embed = global_add_pool(node_embed, branch)

        # Branch value prediction
        branch_value = self.mlp(branch_embed)

        # Aggregate branches for each graph
        indices = torch.arange(len(num_branch_per_graph), device=node_embed.device)
        graph_value = scatter_sum(branch_value, torch.repeat_interleave(indices, num_branch_per_graph), dim=0)

        info['graph_value'] = graph_value
        return info
