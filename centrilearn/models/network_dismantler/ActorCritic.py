"""
Actor-Critic model for network dismantling problem.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from centrilearn.utils.builder import (NETWORK_DISMANTLER, build_backbone,
                                       build_head)


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

    def __init__(
        self,
        backbone_cfg: dict,
        actor_head_cfg: dict = None,
        critic_head_cfg: dict = None,
        num_critics: int = 1,
    ):
        super().__init__()

        # Build backbone
        self.backbone = build_backbone(backbone_cfg)

        # Get backbone output dimension
        if hasattr(self.backbone, "output_dim"):
            self.output_dim = self.backbone.output_dim
        else:
            raise ValueError("Backbone must have 'output_dim' property")

        # Build actor head (policy)
        if actor_head_cfg is None:
            actor_head_cfg = {
                "type": "PolicyHead",
                "in_channels": self.output_dim,
                "hidden_layers": [1],
            }

        self.actor = build_head(actor_head_cfg)

        # Build critic heads (value estimation)
        if critic_head_cfg is None:
            critic_head_cfg = {
                "type": "VHead",
                "in_channels": self.output_dim,
                "hidden_layers": [1],
            }

        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            self.critics.append(build_head(critic_head_cfg))

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            component: Component assignment for each node [num_nodes] (optional)

        Returns:
            logit: Action logits [num_nodes, 1]
            critic_values: List of critic outputs [num_nodes, 1] each
        """
        # Get backbone features
        info = self.backbone(info)
        info = self.actor(info)

        critic_values = [critic(info)["v_values"] for critic in self.critics]
        all_values = torch.stack(critic_values, dim=-1)
        info["v_values"], _ = torch.min(all_values, dim=-1, keepdim=True)

        return info
