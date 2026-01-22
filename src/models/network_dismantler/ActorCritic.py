"""
Actor-Critic model for network dismantling problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.registry import NETWORK_DISMANTLER
from ..utils.builder import build_backbone, build_head


@NETWORK_DISMANTLER.register_module()
class ActorCritic(nn.Module):
    """Actor-Critic model for network dismantling tasks.

    This model combines a graph backbone with actor (policy) and critic (value) heads.
    Supports multiple critics for conservative value estimation.

    Args:
        backbone_cfg: Backbone configuration dictionary.
        actor_head_cfg: Actor head configuration.
        critic_head_cfg: Critic head configuration.
        num_critics: Number of critics for conservative value estimation.
    """

    def __init__(self,
                 backbone_cfg: dict,
                 actor_head_cfg: dict = None,
                 critic_head_cfg: dict = None,
                 num_critics: int = 1):
        super().__init__()

        # Build backbone
        self.backbone = build_backbone(backbone_cfg)

        # Get backbone output dimension
        if hasattr(self.backbone, 'out_channels'):
            self.out_channels = self.backbone.out_channels
        else:
            raise ValueError("Backbone must have 'out_channels' property")

        # Build actor head (policy)
        if actor_head_cfg is None:
            actor_head_cfg = {
                'type': 'LogitHead',
                'in_channels': self.out_channels,
                'hidden_layers': [1],
            }

        self.actor = build_head(actor_head_cfg)

        # Build critic heads (value estimation)
        if critic_head_cfg is None:
            critic_head_cfg = {
                'type': 'VHead',
                'in_channels': self.out_channels,
                'hidden_layers': [1],
            }

        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            self.critics.append(build_head(critic_head_cfg))

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            branch: Branch assignment for each node [num_nodes] (optional)
            **kwargs: optional arguments for GraphSAGE.

        Returns:
            If branch is None:
                logit: Action logits [num_nodes, 1]
                critic_values: List of critic outputs [num_nodes, 1] each
                node_embed: Node embeddings [num_nodes, out_channels]
                graph_embed: Graph embeddings [batch_size, out_channels]

            If branch is not None:
                logit: Action logits [num_nodes, 1]
                value: Aggregated values [batch_size, 1]
                node_embed: Node embeddings [num_nodes, out_channels]
                graph_embed: Graph embeddings [batch_size, out_channels]
        """
        # Get backbone features
        node_embed, graph_embed = self.backbone(x, edge_index, batch, **kwargs)

        # Compute actor logits
        logit = self.actor(node_embed, batch, graph_embed, **kwargs)

        # Compute critic values
        critic_values = [critic(graph_embed, batch, **kwargs) for critic in self.critics]

        all_values = torch.stack(critic_values, dim=-1)
        value, _ = torch.min(all_values, dim=-1, keepdim=True)
        
        # Return as dictionary for clarity
        return {
            'logit': logit,
            'value': value,
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
                - branch: Branch assignment for each node [num_nodes] (optional)
                - Other optional keys

        Returns:
            Dictionary containing:
                - logit: Action logits [num_nodes, 1]
                - value: Aggregated values or list of critic values
                - node_embed: Node embeddings [num_nodes, out_channels]
                - graph_embed: Graph embeddings [batch_size, out_channels]
                - Original info keys
        """
        # Get backbone features
        info = self.backbone.forward_info(info)

        # Compute actor logits
        info = self.actor.forward_info(info)
 
        # Compute critic values
        critic_values = []
        for critic in self.critics:
            critic_info = critic.forward_info(info.copy())
            critic_values.append(critic_info['value'])

        critic_values = torch.stack(critic_values, dim=-1)
        value, _ = torch.min(critic_values, dim=-1)

        info['value'] = value

        return info

    def get_action(self, x, edge_index, batch):
        """Get action from policy (for inference).

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            action: Action indices
            logit: Action logits
        """
        with torch.no_grad():
            output = self.forward(x, edge_index, batch)
            logit = output['logit']
            action_probs = F.softmax(logit, dim=0)
            action = torch.multinomial(action_probs, 1)
        return action, logit

    def get_value(self, x, edge_index, batch, branch):
        """Get value estimation.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            branch: Branch assignment

        Returns:
            value: Value estimation
        """
        with torch.no_grad():
            output = self.forward(x, edge_index, batch, branch)
            return output['value']
