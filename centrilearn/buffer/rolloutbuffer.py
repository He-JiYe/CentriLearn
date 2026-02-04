"""
轨迹缓冲区
"""

from typing import Any, Dict, List

import numpy as np
import torch

from centrilearn.utils.registry import REPLAYBUFFERS


@REPLAYBUFFERS.register_module()
class RolloutBuffer:
    """PPO 轨迹缓冲区
    存储 PPO 算法的完整轨迹，支持 GAE 优势计算和批次生成。
    """

    def __init__(self, capacity: int):
        """初始化轨迹缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(
        self,
        state: Dict[str, Any],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ):
        """添加经验

        Args:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作对数概率
            reward: 获得的奖励
            done: 是否终止
            value: 状态价值估计
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_batches(
        self, batch_size: int, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> List[Dict[str, torch.Tensor]]:
        """获取训练批次，计算 GAE 优势

        Args:
            batch_size: 批次大小
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数

        Returns:
            批次数据列表
        """
        num_steps = len(self.states)

        # 计算 returns
        returns = self._compute_returns(gamma)

        # 计算 GAE 优势
        advantages = self._compute_advantages(gamma, gae_lambda, returns)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备批次
        batches = []
        indices = np.arange(num_steps)
        np.random.shuffle(indices)

        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            batch_indices = indices[start:end]

            batch = {
                "states": [self.states[i] for i in batch_indices],
                "actions": [self.actions[i] for i in batch_indices],
                "old_log_probs": [self.log_probs[i] for i in batch_indices],
                "returns": [returns[i] for i in batch_indices],
                "advantages": [advantages[i] for i in batch_indices],
                "old_values": [self.values[i] for i in batch_indices],
            }
            batches.append(batch)

        return batches

    def _compute_returns(self, gamma: float) -> torch.Tensor:
        """计算折扣回报

        Args:
            gamma: 折扣因子

        Returns:
            折扣回报
        """
        returns = torch.zeros(len(self.rewards))
        running_return = 0

        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + gamma * running_return * (
                1 - self.dones[t]
            )
            returns[t] = running_return

        return returns

    def _compute_advantages(
        self, gamma: float, gae_lambda: float, returns: torch.Tensor
    ) -> torch.Tensor:
        """计算 GAE 优势

        Args:
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            returns: 折扣回报

        Returns:
            GAE 优势
        """
        advantages = torch.zeros(len(returns))
        running_advantage = 0

        values = torch.tensor(self.values).squeeze(-1)

        for t in reversed(range(len(returns))):
            if t == len(returns) - 1:
                next_value = 0
                next_non_terminal = 1 - self.dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values[t]
            running_advantage = (
                delta + gamma * gae_lambda * next_non_terminal * running_advantage
            )
            advantages[t] = running_advantage

        return advantages

    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)
