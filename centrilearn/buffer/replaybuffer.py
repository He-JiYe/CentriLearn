"""
Experience Replay Buffer
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from centrilearn.utils.registry import REPLAYBUFFERS


@REPLAYBUFFERS.register_module()
class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(
        self,
        capacity: int,
        n_step: int = 1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        prioritized: bool = False,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Buffer capacity
            n_step: Number of steps for N-step return
            gamma: Discount factor
            alpha: Priority exponent (0 for uniform sampling, 1 for pure priority-based)
            beta_start: Initial beta for importance sampling correction
            beta_frames: Number of frames for beta to increase from beta_start to 1
            epsilon: Small value added to priorities to avoid zero priority
            prioritized: Whether to use prioritized sampling
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.prioritized = prioritized

        if self.prioritized:
            self.alpha = alpha
            self.beta_start = beta_start
            self.beta_frames = beta_frames
            self.epsilon = epsilon
            self.beta = beta_start
            self.frame = 0
            self.max_priority = 1.0
            self.priorities = np.zeros(capacity)
            self.buffer = [None] * capacity  # Initialize as fixed-size list
            self.pos = 0
            self.size = 0
        else:
            self.buffer = deque(maxlen=capacity)

        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def push(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        reward: float,
        done: bool,
    ):
        """Add experience.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
        """
        # Add to N-step buffer
        self.n_step_buffer.append((state, action, next_state, reward, done))

        # If N-step buffer is full, compute N-step return
        if len(self.n_step_buffer) >= self.n_step:
            (
                n_step_state,
                n_step_action,
                n_step_next_state,
                n_step_reward,
                n_step_done,
            ) = self._get_n_step_experience()

            if self.prioritized:
                # Compute max priority (initial priority for new experiences)
                self.priorities[self.pos] = self.max_priority
                self.buffer[self.pos] = (
                    n_step_state,
                    n_step_action,
                    n_step_next_state,
                    n_step_reward,
                    n_step_done,
                )
                self.pos = (self.pos + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)
            else:
                self.buffer.append(
                    (
                        n_step_state,
                        n_step_action,
                        n_step_next_state,
                        n_step_reward,
                        n_step_done,
                    )
                )

    def _get_n_step_experience(self) -> Tuple:
        """Compute N-step return.

        Returns:
            N-step experience tuple
        """
        # Compute cumulative discounted reward
        n_step_reward = 0
        for i in range(len(self.n_step_buffer)):
            _, _, _, reward, done = self.n_step_buffer[i]
            n_step_reward += (self.gamma**i) * reward
            if done:
                break

        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, next_state, _, done = self.n_step_buffer[-1]

        return state, action, next_state, n_step_reward, done

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Tuple], Optional[np.ndarray], Optional[np.ndarray]]:
        """Random sampling.

        Args:
            batch_size: Batch size

        Returns:
            Sampled experience list
            If using prioritized sampling, also returns:
            - indices: Sampled indices
            - weights: Importance sampling weights
        """
        if self.prioritized:
            # Update beta
            self.frame += 1
            self.beta = min(
                1.0,
                self.beta_start
                + self.frame * (1.0 - self.beta_start) / self.beta_frames,
            )

            # Compute sampling probabilities
            priorities = self.priorities[: self.size]
            probs = priorities**self.alpha
            probs /= probs.sum()

            # Sample
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize

            samples = [self.buffer[i] for i in indices]
            return samples, indices, weights
        else:
            samples = random.sample(self.buffer, batch_size)
            return samples, None, None

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities.

        Args:
            indices: Indices to update
            priorities: New priority values
        """
        if not self.prioritized:
            return

        # Add epsilon and apply alpha
        priorities = (priorities + self.epsilon) ** self.alpha

        # Update priorities
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def get_beta(self) -> float:
        """Get current beta value."""
        return self.beta

    def __len__(self) -> int:
        """Get buffer size."""
        if self.prioritized:
            return self.size
        return len(self.buffer)

    def clear(self):
        """Clear buffer."""
        if self.prioritized:
            self.priorities = np.zeros(self.capacity)
            self.pos = 0
            self.size = 0
            self.max_priority = 1.0
        else:
            self.buffer.clear()

    def __getitem__(self, idx: int):
        """Get experience by index."""
        if self.prioritized:
            return self.buffer[idx]
        return self.buffer[idx]
