"""
Rollout Buffer for PPO Algorithm
"""

from typing import Any, Dict, List

import torch

from centrilearn.utils.registry import REPLAYBUFFERS


@REPLAYBUFFERS.register_module()
class RolloutBuffer:
    """PPO Rollout Buffer for storing trajectories and computing GAE advantages."""

    def __init__(self):
        """Initialize an empty rollout buffer."""
        self.states = []
        self.actions = []
        self.next_state = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def push(
        self,
        state: Dict[str, Any],
        action: torch.Tensor,
        next_state: Dict[str, Any],
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ):
        """Add an experience to the buffer.

        Args:
            state: Current state
            action: Executed action
            next_state: Next state
            reward: Received reward
            done: Whether episode is done
            log_prob: Log probability of the action
            value: Estimated state value
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_state.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_batches(self) -> Dict[str, List]:
        """Get stored batches for training.

        Returns:
            Dictionary containing states, actions, rewards, etc.
        """
        if len(self.states) == 0:
            return {}

        return {
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_state,
            "rewards": self.rewards,
            "dones": self.dones,
            "old_log_probs": self.log_probs,
        }

    def clear(self):
        """Clear all stored experiences."""
        self.states.clear()
        self.actions.clear()
        self.next_state.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        """Return the number of stored experiences."""
        return len(self.states)
