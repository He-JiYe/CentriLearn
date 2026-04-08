"""
Proximal Policy Optimization (PPO) Implementation
For discrete action space tasks like network dismantling
"""

import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_log_softmax, scatter_mean, scatter_softmax
from tqdm import tqdm

from centrilearn.algorithms.base import BaseAlgorithm
from centrilearn.models.loss import reconstruction_loss
from centrilearn.utils import ALGORITHMS, build_network_dismantler


@ALGORITHMS.register_module()
class PPO(BaseAlgorithm):
    """Proximal Policy Optimization with Actor-Critic architecture.

    Args:
        model_cfg: Actor-Critic model configuration
        optimizer_cfg: Optimizer configuration
        replaybuffer_cfg: Rollout buffer configuration
        algo_cfg: PPO algorithm configuration (gamma, gae_lambda, clip_epsilon, etc.)
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
        """Initialize PPO algorithm."""
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.gae_lambda = algo_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = algo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = algo_cfg.get("entropy_coef", 0.01)
        self.value_coef = algo_cfg.get("value_coef", 1)
        self.rcst_coef = algo_cfg.get("rcst_coef", 0.0001)
        self.max_grad_norm = algo_cfg.get("max_grad_norm", 1)
        self.num_epochs = algo_cfg.get("num_epochs", 2)

        super().__init__(
            model_cfg,
            optimizer_cfg,
            scheduler_cfg,
            replaybuffer_cfg,
            metric_manager_cfg,
            device,
        )

    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """Build model from configuration.

        Args:
            model_cfg: Model configuration dictionary

        Returns:
            Built model instance
        """
        return build_network_dismantler(model_cfg)

    def select_action(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """Select action based on current policy.

        Args:
            state: Current state
            **kwargs: Algorithm-specific parameters (e.g., deterministic)

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: Estimated state value
        """
        deterministic = kwargs.get("deterministic", False)
        self.set_eval_mode()

        with torch.no_grad():
            info = state["pyg_data"]
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

            logit = output["logit"].view(-1)
            value = output["v_values"].view(-1)

            if deterministic:
                action = torch.argmax(logit, dim=0)
                log_prob = F.log_softmax(logit, dim=0)[action]
            else:
                probs = F.softmax(logit, dim=0)
                action = torch.multinomial(probs, 1).squeeze(0)
                log_prob = F.log_softmax(logit, dim=0)[action]

        return action.item(), log_prob.item(), value.item()

    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """Collect experience to rollout buffer.

        Args:
            state: Current state
            *args: Other required args (action, next_state, reward, done, log_prob, value)
            **kwargs: Optional args
        """
        action, next_state, reward, done, log_prob, value = args
        self.replay_buffer.push(
            state, action, next_state, reward, done, log_prob, value
        )

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update the model with collected experiences.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of training metrics
        """
        batches = self.replay_buffer.get_batches()

        self.set_train_mode()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_rcst_loss = 0
        total_grad = 0
        num_updates = 0

        states = Batch.from_data_list([i["pyg_data"] for i in batches["states"]]).to(
            self.device
        )
        actions = torch.as_tensor(
            batches["actions"], dtype=torch.long, device=self.device
        )
        next_states = Batch.from_data_list(
            [i["pyg_data"] for i in batches["next_states"]]
        ).to(self.device)
        rewards = torch.as_tensor(
            batches["rewards"], dtype=torch.float, device=self.device
        )
        dones = torch.as_tensor(batches["dones"], dtype=torch.float, device=self.device)
        old_log_probs = torch.as_tensor(
            batches["old_log_probs"], dtype=torch.float, device=self.device
        )

        transitions_length = len(batches["states"])

        for _ in range(self.num_epochs):
            with torch.no_grad():
                values = self.model(
                    {
                        "x": states.x,
                        "edge_index": states.edge_index,
                        "batch": states.get(
                            "batch",
                            torch.zeros(
                                states.x.shape[0], dtype=torch.long, device=self.device
                            ),
                        ),
                        "component": states.get("component"),
                    }
                )["v_values"].view(-1)
                next_values = self.model(
                    {
                        "x": next_states.x,
                        "edge_index": next_states.edge_index,
                        "batch": next_states.get(
                            "batch",
                            torch.zeros(
                                next_states.x.shape[0],
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ),
                        "component": next_states.get("component"),
                    }
                )["v_values"].view(-1)

                td_targets = rewards + self.gamma * next_values * (1 - dones)
                td_errors = td_targets - values

                advantage_list = []
                advantage = 0.0
                for delta, done in zip(td_errors.flip(0), dones.flip(0)):
                    advantage = (
                        self.gamma * self.gae_lambda * advantage * (1 - done) + delta
                    )
                    advantage_list.append(advantage)
                advantages = torch.stack(advantage_list[::-1]).to(td_errors.device)

            indices = torch.randperm(transitions_length).to(self.device)
            for start in range(0, transitions_length, batch_size):
                end = start + batch_size
                batch = indices[start:end]

                states_b = Batch.from_data_list(
                    [batches["states"][i]["pyg_data"] for i in batch]
                ).to(self.device)
                actions_b = actions[batch]
                old_log_probs_b = old_log_probs[batch]
                advantage_b = advantages[batch]
                td_target_b = td_targets[batch]

                actions_b = actions_b + states_b.ptr[:-1]

                batch_indices = states_b.get(
                    "batch", torch.zeros(states_b.x.shape[0], dtype=torch.long)
                )
                output = self.model(
                    {
                        "x": states_b.x,
                        "edge_index": states_b.edge_index,
                        "batch": batch_indices,
                        "component": states_b.get("component"),
                    }
                )
                new_logit = output["logit"].view(-1)
                new_value = output["v_values"].view(-1)

                log_prob = scatter_log_softmax(new_logit, batch_indices, dim=0)
                ratio = torch.exp(log_prob[actions_b] - old_log_probs_b)

                surr1 = ratio * advantage_b
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantage_b
                )
                policy_loss_epoch = -torch.min(surr1, surr2).mean()

                value_loss_epoch = F.smooth_l1_loss(new_value, td_target_b)

                probs = scatter_softmax(new_logit, batch_indices, dim=0)
                entropy_loss_epoch = -scatter_mean(
                    probs * log_prob, batch_indices
                ).mean()

                rcst_loss_epoch = reconstruction_loss(
                    output["node_embed"],
                    output["edge_index"],
                    states_b.ptr,
                    device=self.device,
                )

                policy_loss = policy_loss_epoch
                value_loss = self.value_coef * value_loss_epoch
                entropy_loss = -self.entropy_coef * entropy_loss_epoch
                rcst_loss = self.rcst_coef * rcst_loss_epoch
                total_loss = policy_loss + value_loss * 10 + entropy_loss + rcst_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_rcst_loss += rcst_loss.item()
                total_grad += grad.item()
                num_updates += 1

        self.training_step += num_updates
        self.replay_buffer.clear()

        return {
            "policy_loss": total_policy_loss / num_updates if num_updates > 0 else 0.0,
            "value_loss": total_value_loss / num_updates if num_updates > 0 else 0.0,
            "entropy_loss": (
                total_entropy_loss / num_updates if num_updates > 0 else 0.0
            ),
            "rcst_loss": total_rcst_loss / num_updates if num_updates > 0 else 0.0,
            "grad": total_grad / num_updates if num_updates > 0 else 0.0,
        }

    def learn(
        self,
        env: Any,
        training_cfg: Dict[str, Any],
        verbose: bool = True,
        logger=None,
        tensorboard_writer=None,
    ) -> Dict[str, Any]:
        """PPO training loop.

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
        num_episodes = training_cfg.get("num_episodes", 5000)
        num_update = training_cfg.get("num_update", 10)
        max_steps = training_cfg.get("max_steps", 1000)
        batch_size = training_cfg.get("batch_size", 64)
        log_interval = training_cfg.get("log_interval", 100)
        save_interval = training_cfg.get("save_interval", 1000)
        save_path = training_cfg.get("save_path", None)

        all_rewards = []
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_rcst_losses = []
        all_grads = []

        self.metric_manager.start_timer()

        pbar = tqdm(total=num_episodes, desc="PPO Training", disable=not verbose)
        for episode in range(1, num_episodes + 1):
            env.reset()
            episode_reward = 0.0

            for step in range(max_steps):
                state = env.get_state(use_gcc=use_gcc)
                action, log_prob, value = self.select_action(state)
                reward, done, info = env.step(action, state["mapping"])
                next_state = env.get_state(mask=state["node_mask"])

                self.collect_experience(
                    state, action, next_state, reward, done, log_prob, value
                )
                episode_reward += reward

                if self.metric_manager:
                    self.metric_manager.update(
                        state, action, next_state, reward, done, info
                    )

                if done or step >= max_steps:
                    all_rewards.append(episode_reward)
                    break

            if episode % num_update == 0:
                metrics = self.update(batch_size)
                all_policy_losses.append(metrics.get("policy_loss", 0.0))
                all_value_losses.append(metrics.get("value_loss", 0.0))
                all_entropy_losses.append(metrics.get("entropy_loss", 0.0))
                all_rcst_losses.append(metrics.get("rcst_loss", 0.0))
                all_grads.append(metrics.get("grad", 0.0))

                if tensorboard_writer:
                    tensorboard_writer.add_scalar(
                        "Train/reward", episode_reward, episode
                    )
                    tensorboard_writer.add_scalar(
                        "Train/policy_loss", all_policy_losses[-1], episode
                    )
                    tensorboard_writer.add_scalar(
                        "Train/value_loss", all_value_losses[-1], episode
                    )
                    tensorboard_writer.add_scalar(
                        "Train/entropy_loss", all_entropy_losses[-1], episode
                    )
                    tensorboard_writer.add_scalar(
                        "Train/rcst_loss", all_rcst_losses[-1], episode
                    )
                    tensorboard_writer.add_scalar("Train/grad", all_grads[-1], episode)

            if self.metric_manager:
                summary = self.metric_manager.get_summary()
            else:
                summary = {}
            pbar.set_postfix(
                {
                    "reward": f"{episode_reward:.2f}",
                    "policy_loss": (
                        f"{all_policy_losses[-1]:.4f}" if all_policy_losses else "0"
                    ),
                    **summary,
                }
            )
            pbar.update(1)

            if episode % log_interval == 0 and all_rewards:
                avg_reward = sum(all_rewards[-log_interval:]) / min(
                    log_interval, len(all_rewards)
                )
                avg_policy_loss = sum(all_policy_losses[-log_interval:]) / min(
                    log_interval, len(all_policy_losses)
                )
                avg_value_loss = sum(all_value_losses[-log_interval:]) / min(
                    log_interval, len(all_value_losses)
                )
                avg_entropy_loss = sum(all_entropy_losses[-log_interval:]) / min(
                    log_interval, len(all_entropy_losses)
                )
                avg_rcst_loss = sum(all_rcst_losses[-log_interval:]) / min(
                    log_interval, len(all_rcst_losses)
                )
                avg_grad = sum(all_grads[-log_interval:]) / min(
                    log_interval, len(all_grads)
                )

                log_msg = (
                    f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                    f"Policy Loss: {avg_policy_loss:.4f}, "
                    f"Value Loss: {avg_value_loss:.4f}, "
                    f"Entropy Loss: {avg_entropy_loss:.4f}, "
                    f"Rcst Loss: {avg_rcst_loss:.4f}, "
                    f"Grad: {avg_grad:.4f}"
                )

                if logger:
                    if self.metric_manager:
                        summary = self.metric_manager.get_summary()
                        metrics_str = ", ".join(
                            [f"{k}: {v:.4f}" for k, v in summary.items()]
                        )
                        log_msg += f", {metrics_str}"
                    logger.info(log_msg)

            # Save checkpoint
            if save_path and episode % save_interval == 0:
                self.save_checkpoint(
                    f"{save_path}/checkpoint_episode_{episode}.pt", episode=episode
                )

        pbar.close()

        if save_path:
            self.save_checkpoint(
                f"{save_path}/checkpoint_final.pt", episode=num_episodes
            )

            if verbose:
                print(f"Final model saved: {save_path}/checkpoint_final.pt")
            if logger:
                logger.info(f"Final model saved: {save_path}/checkpoint_final.pt")

        return {
            "num_episodes": num_episodes,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
            "avg_reward": (
                sum(all_rewards[-100:]) / min(100, len(all_rewards))
                if all_rewards
                else 0.0
            ),
            "training_steps": self.training_step,
        }

    def rollout(self, env, use_gcc: bool = False, attack_rate_per_step: float = 0.01):
        """Execute one complete rollout episode."""
        num_nodes = env.num_nodes
        attack_times = max(int(num_nodes * attack_rate_per_step), 1)

        self.set_eval_mode()
        start = time.time()
        env.reset(use_gcc=use_gcc)
        done = False
        while not done:
            # Compute policy
            state = env.get_state(use_gcc=use_gcc)
            with torch.no_grad():
                info = state["pyg_data"]
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

            logit = output["logit"].view(-1)
            indices = logit.argsort(descending=True)

            # Start attack
            for action in indices[:attack_times]:
                _, done, _ = env.step(action, state["mapping"])

                if done:
                    break
        end = time.time()

        return {**env.rollout_info(), "rollout_time": end - start}
