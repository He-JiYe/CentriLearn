"""
Deep Q-Network (DQN) Implementation
"""

import copy
import random
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_max
from tqdm import tqdm

from centrilearn.algorithms.base import BaseAlgorithm
from centrilearn.models.loss import reconstruction_loss
from centrilearn.utils import ALGORITHMS, build_network_dismantler


@ALGORITHMS.register_module()
class DQN(BaseAlgorithm):
    """Deep Q-Network with experience replay and target network.

    Args:
        model_cfg: Q-network model configuration
        optimizer_cfg: Optimizer configuration
        replaybuffer_cfg: Experience replay buffer configuration
        algo_cfg: DQN algorithm configuration (gamma, epsilon, etc.)
        scheduler_cfg: Learning rate scheduler configuration (optional)
        metric_manager_cfg: Metric manager configuration (optional)
        device: Device to run on
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        replaybuffer_cfg: Optional[Dict[str, Any]] = None,
        algo_cfg: Optional[Dict[str, Any]] = {},
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        metric_manager_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """Initialize DQN algorithm."""
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.epsilon_start = algo_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = algo_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = algo_cfg.get("epsilon_decay", 10000)
        self.tau = algo_cfg.get("tau", 0.005)
        self.grad_norm = algo_cfg.get("grad_norm", 1.0)
        self.rcst_coef = algo_cfg.get("rcst_coef", 0.0001)

        super().__init__(
            model_cfg,
            optimizer_cfg,
            scheduler_cfg,
            replaybuffer_cfg,
            metric_manager_cfg,
            device,
        )

        self.target_model = self._build_model(model_cfg).to(device)
        self.target_model.eval()
        self.target_model.load_state_dict(self.model.state_dict())

    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """Build model from configuration.

        Args:
            model_cfg: Model configuration dictionary

        Returns:
            Built model instance
        """
        return build_network_dismantler(model_cfg)

    def compute_epsilon(self) -> float:
        """Compute current exploration rate (epsilon) for epsilon-greedy policy."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.training_step / self.epsilon_decay
        )

    def select_action(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            **kwargs: Algorithm-specific parameters (e.g., epsilon)

        Returns:
            action: Selected action
            epsilon: Exploration rate used
        """
        epsilon = kwargs.get("epsilon", None)
        if epsilon is None:
            epsilon = self.compute_epsilon()

        if random.random() < epsilon:
            num_nodes = state["pyg_data"].x.shape[0]
            action = torch.randint(0, num_nodes, (1,))
        else:
            with torch.no_grad():
                info = state["pyg_data"].to(self.device)
                output = self.model(
                    {
                        "x": info.x,
                        "edge_index": info.edge_index,
                        "batch": info.get(
                            "batch",
                            torch.zeros(
                                info.x.shape[0], dtype=torch.long, device=self.device
                            ),
                        ),
                        "component": info.get("component"),
                    }
                )
                q_values = output["q_values"].squeeze(-1)
                action = torch.argmax(q_values)

        return action.item(), epsilon

    def update(self, batch_size: int) -> Dict[str, float]:
        """Update the model with a batch of experiences.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0}

        batch, indices, weights = self.replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        self.set_train_mode()

        state_info = Batch.from_data_list([state["pyg_data"] for state in states]).to(
            self.device
        )
        next_state_info = Batch.from_data_list(
            [next_state["pyg_data"] for next_state in next_states]
        ).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)

        actions = actions + state_info.ptr[:-1]

        with torch.set_grad_enabled(True):
            state_batch = state_info.get(
                "batch", torch.zeros(state_info.x.shape[0], dtype=torch.long)
            )
            output = self.model(
                {
                    "x": state_info.x,
                    "edge_index": state_info.edge_index,
                    "batch": state_batch,
                    "component": state_info.get("component"),
                }
            )
            current_q_values = output["q_values"].squeeze(-1)[actions]

        with torch.no_grad():
            next_state_batch = next_state_info.get(
                "batch",
                torch.zeros(
                    next_state_info.x.shape[0], dtype=torch.long, device=self.device
                ),
            )
            next_output = self.target_model(
                {
                    "x": next_state_info.x,
                    "edge_index": next_state_info.edge_index,
                    "batch": next_state_batch,
                    "component": next_state_info.get("component"),
                }
            )
            next_q_values = scatter_max(
                next_output["q_values"].squeeze(-1), next_state_batch
            )[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        rcst_loss = reconstruction_loss(
            output["node_embed"],
            output["edge_index"],
            state_info.ptr,
            device=self.device,
        )
        total_loss = loss + rcst_loss * self.rcst_coef

        if weights is not None:
            weights_tensor = torch.as_tensor(
                weights, dtype=torch.float, device=self.device
            )
            total_loss = (total_loss * weights_tensor).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_norm
        )
        self.optimizer.step()

        if indices is not None and weights is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q_values - target_q_values)
                priorities = td_errors.cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)

        self._update_target_network()
        self.training_step += 1

        return {
            "loss": total_loss.item(),
            "grad": grad.item(),
            "epsilon": self.compute_epsilon(),
            "training_step": self.training_step,
        }

    def _update_target_network(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """Collect experience to replay buffer.

        Args:
            state: Current state
            *args: Other required args (action, next_state, reward, done)
            **kwargs: Optional args
        """
        action, next_state, reward, done = args[:4]
        self.replay_buffer.push(state, action, next_state, reward, done)
        _ = kwargs

    def get_q_values(self, state: Dict[str, Any]) -> torch.Tensor:
        """Get Q values for state.

        Args:
            state: State

        Returns:
            Q values
        """
        self.set_eval_mode()
        with torch.no_grad():
            info = state["pyg_data"]
            output = self.model(
                {
                    "x": info.x,
                    "edge_index": info.edge_index,
                    "batch": info.get(
                        "batch", torch.zeros(info.x.shape[0], dtype=torch.long)
                    ),
                    "component": info.component,
                }
            )
            return output["q_values"].squeeze(-1)

    def learn(
        self,
        env: Any,
        training_cfg: Dict[str, Any],
        verbose: bool = True,
        logger=None,
        tensorboard_writer=None,
    ) -> Dict[str, Any]:
        """DQN training loop.

        Args:
            env: Environment instance
            training_cfg: Training configuration
            verbose: Whether to print logs
            logger: Logger object
            tensorboard_writer: TensorBoard writer object

        Returns:
            Training results dictionary
        """

        assert self.optimizer is not None, f"Training failed: optimizer is missing."
        assert (
            self.replay_buffer is not None
        ), f"Training failed: replay buffer is missing."

        use_gcc = training_cfg.get("use_gcc", False)
        num_episodes = training_cfg.get("num_episodes", 3000)
        batch_size = training_cfg.get("batch_size", 32)
        max_steps = training_cfg.get("max_steps", 1000)
        log_interval = training_cfg.get("log_interval", 10)
        save_interval = training_cfg.get("save_interval", 100)
        save_path = training_cfg.get("save_path", None)

        state = env.reset(use_gcc=use_gcc)

        all_rewards = []
        all_losses = []
        all_grads = []

        self.metric_manager.start_timer()

        pbar = tqdm(total=num_episodes, desc="DQN Training", disable=not verbose)
        for episode in range(1, num_episodes + 1):
            env.reset()
            episode_reward, episode_losses, episode_grads = 0.0, 0.0, 0.0

            for step in range(max_steps):
                state = env.get_state(use_gcc=use_gcc)
                epsilon = self.compute_epsilon()
                action, _ = self.select_action(state, epsilon=epsilon)
                reward, done, info = env.step(action, state["mapping"])
                next_state = env.get_state(mask=state["node_mask"])

                self.collect_experience(state, action, next_state, reward, done)
                episode_reward += reward

                if self.metric_manager:
                    self.metric_manager.update(
                        state, action, next_state, reward, done, info
                    )

                metrics = self.update(batch_size)
                episode_losses += metrics.get("loss", 0.0)
                episode_grads += metrics.get("grad", 0.0)

                if done or step >= max_steps:
                    all_rewards.append(episode_reward)
                    all_losses.append(episode_losses)
                    all_grads.append(episode_grads)
                    break

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Train/reward", episode_reward, episode)
                tensorboard_writer.add_scalar("Train/loss", episode_losses, episode)
                tensorboard_writer.add_scalar("Train/grad", episode_grads, episode)
                tensorboard_writer.add_scalar("Train/epsilon", epsilon, episode)

            if self.metric_manager:
                summary = self.metric_manager.get_summary()
            else:
                summary = {}
            pbar.set_postfix(
                {
                    "reward": f"{episode_reward:.2f}",
                    "loss": f"{episode_losses:.4f}",
                    "eps": f"{self.compute_epsilon():.3f}",
                    **summary,
                }
            )
            pbar.update(1)

            if episode % log_interval == 0 and all_rewards:
                avg_reward = sum(all_rewards[-log_interval:]) / min(
                    log_interval, len(all_rewards)
                )
                avg_loss = sum(all_losses[-log_interval:]) / min(
                    log_interval, len(all_losses)
                )
                avg_grad = sum(all_grads[-log_interval:]) / min(
                    log_interval, len(all_grads)
                )
                log_msg = (
                    f"Episode {episode}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Avg Grad: {avg_grad:.4f}, "
                    f"Epsilon: {self.compute_epsilon():.3f}"
                )

                if logger:
                    if self.metric_manager:
                        summary = self.metric_manager.get_summary()
                        metrics_str = ", ".join(
                            [f"{k}: {v:.4f}" for k, v in summary.items()]
                        )
                        log_msg += f", {metrics_str}"
                    logger.info(log_msg)

            if save_path and episode % save_interval == 0:
                self.save_checkpoint(
                    f"{save_path}/checkpoint_episode_{episode}.pt",
                    episode=episode,
                )

        pbar.close()

        if save_path:
            self.save_checkpoint(
                f"{save_path}/checkpoint_final.pt", episode=num_episodes
            )

            if verbose:
                print(f"最终模型保存地址: {save_path}/checkpoint_final.pt")
            if logger:
                logger.info(f"最终模型保存地址: {save_path}/checkpoint_final.pt")

        return {
            "num_episodes": num_episodes,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
            "avg_reward": (
                sum(all_rewards[-100:]) / min(100, len(all_rewards))
                if all_rewards
                else 0.0
            ),
            "total_steps": self.training_step,
        }

    def rollout(self, env, use_gcc: bool = False, attack_rate_per_step: float = 0.01):
        """Execute one complete rollout episode."""
        num_nodes = env.num_nodes
        attack_times = max(int(num_nodes * attack_rate_per_step), 1)

        self.set_eval_mode()
        start = time.time()
        env.reset()
        done = False
        while not done:
            state = env.get_state(use_gcc=use_gcc)
            with torch.no_grad():
                info = state["pyg_data"].to(self.device)
                output = self.model(
                    {
                        "x": info.x,
                        "edge_index": info.edge_index,
                        "batch": info.get(
                            "batch",
                            torch.zeros(
                                info.x.shape[0], dtype=torch.long, device=self.device
                            ),
                        ),
                        "component": info.get("component"),
                    }
                )
                q_values = output["q_values"].squeeze(-1)
                indices = q_values.argsort(descending=True)

            for action in indices[:attack_times]:
                _, done, _ = env.step(action, state["mapping"])

                if done:
                    break
        end = time.time()

        return {**env.rollout_info(), "rollout_time": end - start}
