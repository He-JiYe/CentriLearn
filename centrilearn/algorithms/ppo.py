"""
Proximal Policy Optimization (PPO) 算法实现
适用于网络瓦解等离散动作空间任务
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_log_softmax, scatter_softmax

from centrilearn.utils import ALGORITHMS, build_network_dismantler

from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class PPO(BaseAlgorithm):
    """Proximal Policy Optimization 算法

    实现 Actor-Critic 架构的 PPO-Clip 算法。

    Args:
        model: Actor-Critic 模型实例 或 模型配置字典
        optimizer_cfg: 优化器配置
        scheduler_cfg: 学习率调度器配置
        replaybuffer_cfg: 轨迹缓冲区配置
        metric_manager_cfg: 指标管理器配置
        algo_cfg: 算法配置
        device: 运行设备
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict[str, Any]],
        optimizer_cfg: Dict[str, Any],
        replaybuffer_cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        metric_manager_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """初始化 PPO 算法"""
        # 超参数
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.gae_lambda = algo_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = algo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = algo_cfg.get("entropy_coef", 0.01)
        self.value_coef = algo_cfg.get("value_coef", 0.5)
        self.max_grad_norm = algo_cfg.get("max_grad_norm", 0.5)
        self.num_epochs = algo_cfg.get("num_epochs", 10)

        # 调用父类初始化（支持模型配置）
        super().__init__(
            model,
            optimizer_cfg,
            scheduler_cfg,
            replaybuffer_cfg,
            metric_manager_cfg,
            device,
        )

    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """从配置构建模型

        Args:
            model_cfg: 模型配置字典

        Returns:
            构建好的模型实例
        """
        return build_network_dismantler(model_cfg)

    def _select_action_single(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """为单个环境选择动作

        Args:
            state: 当前状态
            **kwargs: 算法特定的参数（如 deterministic）

        Returns:
            action: 选择的动作
            log_prob: 动作对数概率
            value: 状态价值（标量）
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

            logit = output["logit"].squeeze()
            value = output["v_values"].squeeze()

            if deterministic:
                # 确定性策略：选择概率最大的动作
                action = torch.argmax(logit, dim=0)
                log_prob = F.log_softmax(logit, dim=0)[action]
            else:
                # 随机策略：按概率采样
                probs = F.softmax(logit, dim=0)
                action = torch.multinomial(probs, 1)
                log_prob = F.log_softmax(logit, dim=0)[action]

        return action.item(), log_prob.item(), value.item()

    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """收集经验到轨迹缓冲区

        Args:
            state: 当前状态
            *args: 其他必需参数（action, log_prob, reward, done, value）
            **kwargs: 可选参数
        """
        action, reward, _, done, log_prob, value = args
        self.replay_buffer.push(state, action, reward, done, log_prob, value)

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """更新模型

        Args:
            batch_size: 批次大小

        Returns:
            训练指标字典
        """
        if len(self.replay_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

        # 获取训练批次
        batches = self.replay_buffer.get_batches(
            batch_size=batch_size, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        self.set_train_mode()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_updates = 0

        # 多轮更新
        for _ in range(self.num_epochs):
            for batch in batches:
                # 前向传播
                policy_loss_epoch = 0
                value_loss_epoch = 0
                entropy_loss_epoch = 0

                state_info = Batch.from_data_list(
                    [i["pyg_data"] for i in batch["states"]]
                ).to(self.device)
                actions = torch.as_tensor(
                    batch["actions"], dtype=torch.long, device=self.device
                )
                old_log_probs = torch.as_tensor(
                    batch["old_log_probs"], dtype=torch.float, device=self.device
                )
                returns = torch.as_tensor(
                    batch["returns"], dtype=torch.float, device=self.device
                )
                advantages = torch.as_tensor(
                    batch["advantages"], dtype=torch.float, device=self.device
                )
                old_values = torch.as_tensor(
                    batch["old_values"], dtype=torch.float, device=self.device
                )

                # 前向传播
                batch_indices = state_info.get(
                    "batch", torch.zeros(state_info.x.shape[0], dtype=torch.long)
                )
                output = self.model(
                    {
                        "x": state_info.x,
                        "edge_index": state_info.edge_index,
                        "batch": batch_indices,
                        "component": state_info.get("component"),
                    }
                )

                new_logit = output["logit"].squeeze()
                new_value = output["v_values"].squeeze()

                # 计算 ratio
                log_prob = scatter_log_softmax(new_logit, batch_indices, dim=0)
                new_log_prob = log_prob[actions]
                ratio = torch.exp(new_log_prob - old_log_probs)

                # 计算 PPO loss
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )
                policy_loss_epoch = -torch.min(surr1, surr2).mean()

                # 计算 value loss
                value_clipped = old_values + torch.clamp(
                    new_value - old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = F.mse_loss(new_value, returns.squeeze(), reduction="none")
                value_loss2 = F.mse_loss(value_clipped, returns, reduction="none")
                value_loss_epoch = torch.max(value_loss1, value_loss2).mean()

                # 计算熵损失
                probs = scatter_softmax(new_logit, batch_indices, dim=0)
                entropy_loss_epoch = -torch.sum(probs * log_prob)

                # 合并 losses
                policy_loss = policy_loss_epoch
                value_loss = self.value_coef * value_loss_epoch
                entropy_loss = -self.entropy_coef * entropy_loss_epoch

                total_loss = policy_loss + value_loss + entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        self.training_step += num_updates

        # 清空缓冲区
        self.replay_buffer.clear()

        return {
            "policy_loss": total_policy_loss / num_updates if num_updates > 0 else 0.0,
            "value_loss": total_value_loss / num_updates if num_updates > 0 else 0.0,
            "entropy_loss": (
                total_entropy_loss / num_updates if num_updates > 0 else 0.0
            ),
            "training_step": self.training_step,
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一步训练（兼容基类接口）

        Args:
            batch: 训练数据批次

        Returns:
            训练指标字典
        """
        return self.update(batch.get("batch_size", 64))

    def get_action_value(
        self, state: Dict[str, Any]
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """获取动作和价值（用于推理）

        Args:
            state: 状态

        Returns:
            action: 选择的动作
            value: 状态价值（标量）
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
                    "component": info.get("component"),
                }
            )
            logit = output["logit"].squeeze()
            value = output["v_values"].squeeze()

            action = torch.argmax(logit, dim=0)
        return action, value
