"""
轨迹缓冲区
"""

from concurrent.futures import ThreadPoolExecutor
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
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
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

        # 处理 values，确保它们是标量值
        values = []
        for v in self.values:
            if isinstance(v, torch.Tensor):
                values.append(v.item())
            else:
                values.append(v)
        values = torch.tensor(values)

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


@REPLAYBUFFERS.register_module()
class VectorizedRolloutBuffer:
    """向量化轨迹缓冲区

    管理多个 RolloutBuffer 实例，支持并行操作多个环境的轨迹缓冲区。

    Attributes:
        buffers: 缓冲区实例列表
        num_envs: 环境数量
        executor: 线程池执行器
    """

    def __init__(self, env_num: int, capacity: int):
        """初始化向量化轨迹缓冲区

        Args:
            env_num: 环境数量
            capacity: 每个缓冲区的容量
        """
        self.num_envs = env_num
        self.capacity = capacity

        # 为每个环境创建一个 RolloutBuffer 实例
        self.buffers = []
        for _ in range(env_num):
            buffer = RolloutBuffer(capacity=capacity)
            self.buffers.append(buffer)

        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=env_num)

    def push(
        self,
        states: List[Dict[str, Any]],
        actions: List[torch.Tensor],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
    ):
        """批量添加经验

        Args:
            states: 当前状态列表
            actions: 执行的动作列表
            log_probs: 动作对数概率列表
            rewards: 获得的奖励列表
            dones: 是否终止列表
            values: 状态价值估计列表
        """
        if len(states) != self.num_envs:
            raise ValueError(
                f"states 长度 {len(states)} 必须等于环境数量 {self.num_envs}"
            )

        # 并行添加经验到每个缓冲区
        def push_env(i):
            self.buffers[i].push(
                state=states[i],
                action=actions[i],
                log_prob=log_probs[i],
                reward=rewards[i],
                done=dones[i],
                value=values[i],
            )

        list(self.executor.map(push_env, range(self.num_envs)))

    def get_batches(
        self, batch_size: int, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> List[Dict[str, torch.Tensor]]:
        """从所有缓冲区获取训练批次，计算 GAE 优势

        Args:
            batch_size: 批次大小
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数

        Returns:
            批次数据列表
        """

        # 并行从每个缓冲区获取批次
        def get_buffer_batches(i):
            return self.buffers[i].get_batches(batch_size, gamma, gae_lambda)

        results = list(self.executor.map(get_buffer_batches, range(self.num_envs)))

        # 合并结果
        all_batches = []
        for batches in results:
            all_batches.extend(batches)

        return all_batches

    def clear(self):
        """清空所有缓冲区"""

        def clear_buffer(buffer):
            buffer.clear()

        list(self.executor.map(clear_buffer, self.buffers))

    def __len__(self) -> int:
        """返回所有缓冲区的总经验数"""
        return sum(len(buffer) for buffer in self.buffers)

    def __repr__(self) -> str:
        return f"VectorizedRolloutBuffer(num_envs={self.num_envs}, capacity={self.capacity})"

    def __del__(self):
        """清理资源，关闭线程池"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
