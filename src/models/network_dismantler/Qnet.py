"""
Q-network model for network dismantling problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.registry import NETWORK_DISMANTLER
from ..utils.builder import build_backbone, build_head


@NETWORK_DISMANTLER.register_module()
class Qnet(nn.Module):
    """Q-network for network dismantling tasks.

    This model combines a graph backbone with a Q-value prediction head.

    Args:
        backbone_cfg: Backbone configuration dictionary.
        q_head_cfg: Q-value head configuration.
    """

    def __init__(self,
                 backbone_cfg: dict,
                 q_head_cfg: dict = None):
        super().__init__()

        # Build backbone
        self.backbone = build_backbone(backbone_cfg)

        # Get backbone output dimension
        if hasattr(self.backbone, 'out_channels'):
            self.out_channels = self.backbone.out_channels
        else:
            raise ValueError("Backbone must have 'out_channels' property")

        # Build Q-value head
        if q_head_cfg is None:
            q_head_cfg = {
                'type': 'QHead',
                'in_channels': self.out_channels,
                'hidden_layers': [1],
            }

        self.q_head = build_head(q_head_cfg)

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            **kwargs: Other optional keys

        Returns:
            Dictionary containing:
                q_values: Q-values for each node [num_nodes, 1]
                node_embed: Node embeddings [num_nodes, out_channels]
                graph_embed: Graph embeddings [batch_size, out_channels]
        """
        # Get backbone features
        node_embed, graph_embed = self.backbone(x, edge_index, batch, **kwargs)

        # Compute Q-values
        q_values = self.q_head(node_embed, batch, graph_embed, **kwargs)

        return {
            'q_values': q_values,
            'node_embed': node_embed,
            'graph_embed': graph_embed
        }

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]
                - Other optional keys

        Returns:
            Dictionary containing:
                - q_values: Q-values for each node [num_nodes, 1]
                - node_embed: Node embeddings [num_nodes, out_channels]
                - graph_embed: Graph embeddings [batch_size, out_channels]
                - Original info keys
        """
        # Get backbone features using forward_info
        info = self.backbone.forward_info(info)

        # Compute Q-values using forward_info
        info = self.q_head.forward_info(info)

        return info

    def get_q_values(self, x, edge_index, batch):
        """Get Q-values for action selection.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            q_values: Q-values
        """
        with torch.no_grad():
            output = self.forward(x, edge_index, batch)
            return output['q_values']

    def select_action(self, x, edge_index, batch, epsilon=0.0):
        """Select action using epsilon-greedy policy.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            epsilon: Exploration rate

        Returns:
            action: Action index
            q_value: Q-value of selected action
        """
        q_values = self.get_q_values(x, edge_index, batch)

        # Epsilon-greedy action selection
        if torch.rand(1).item() < epsilon:
            # Random action
            action = torch.randint(0, len(q_values), (1,)).squeeze()
            q_value = q_values[action]
        else:
            # Greedy action
            action = torch.argmax(q_values, dim=0)
            q_value = q_values[action]

        return action, q_value

    def compute_loss(self, q_values, target_q_values, weights=None):
        """Compute Q-learning loss.

        Args:
            q_values: Predicted Q-values
            target_q_values: Target Q-values
            weights: Optional sample weights

        Returns:
            loss: Q-learning loss
        """
        loss = F.mse_loss(q_values, target_q_values, reduction='none')

        if weights is not None:
            loss = loss * weights

        return loss.mean()
