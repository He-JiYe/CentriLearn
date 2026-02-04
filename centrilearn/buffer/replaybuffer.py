"""
经验回放缓冲区
"""
from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from collections import deque
from centrilearn.utils.registry import REPLAYBUFFERS

@REPLAYBUFFERS.register_module()
class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self,
                 capacity: int,
                 n_step: int = 1,
                 gamma: float = 0.99,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 epsilon: float = 1e-6,
                 prioritized: bool = False):
        """初始化回放缓冲区

        Args:
            capacity: 缓冲区容量
            n_step: N-step 回报的步数
            gamma: 折扣因子
            alpha: 优先度指数 (0 表示均匀采样，1 表示完全基于优先度)
            beta_start: 重要性采样校正的初始 beta
            beta_frames: beta 从 beta_start 到 1 的帧数
            epsilon: 添加到优先度的最小值，防止零优先度
            prioritized: 是否使用优先度采样
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
            self.buffer = [None] * capacity  # 初始化为固定大小列表
            self.pos = 0
            self.size = 0
        else:
            self.buffer = deque(maxlen=capacity)

        # N-step 缓冲区
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self,
             state: Dict[str, Any],
             action: int,
             reward: float,
             next_state: Dict[str, Any],
             done: bool):
        """添加经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 添加到 N-step 缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # 如果 N-step 缓冲区已满，计算 N-step 回报
        if len(self.n_step_buffer) >= self.n_step:
            n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = \
                self._get_n_step_experience()

            if self.prioritized:
                # 计算最大优先度（新经验的初始优先度）
                self.priorities[self.pos] = self.max_priority
                self.buffer[self.pos] = (n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
                self.pos = (self.pos + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)
            else:
                self.buffer.append((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))

    def _get_n_step_experience(self) -> Tuple:
        """计算 N-step 回报

        Returns:
            N-step 经验元组
        """
        # 计算累积折扣奖励
        n_step_reward = 0
        for i in range(len(self.n_step_buffer)):
            _, _, reward, _, done = self.n_step_buffer[i]
            n_step_reward += (self.gamma ** i) * reward
            if done:
                break

        # 获取初始状态和动作
        state, action, _, _, _ = self.n_step_buffer[0]

        # 获取最后的下一状态
        _, _, _, next_state, done = self.n_step_buffer[-1]

        return state, action, n_step_reward, next_state, done

    def sample(self, batch_size: int) -> Tuple[List[Tuple], Optional[np.ndarray], Optional[np.ndarray]]:
        """随机采样

        Args:
            batch_size: 批次大小

        Returns:
            采样的经验列表
            如果使用优先度采样，还返回：
            - indices: 采样的索引
            - weights: 重要性采样权重
        """
        if self.prioritized:
            # 更新 beta
            self.frame += 1
            self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

            # 计算采样概率
            priorities = self.priorities[:self.size]
            probs = priorities ** self.alpha
            probs /= probs.sum()

            # 采样
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # 归一化

            samples = [self.buffer[i] for i in indices]
            return samples, indices, weights
        else:
            samples = random.sample(self.buffer, batch_size)
            return samples, None, None

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先度

        Args:
            indices: 需要更新的索引
            priorities: 新的优先度值
        """
        if not self.prioritized:
            return

        # 添加 epsilon 并应用 alpha
        priorities = (priorities + self.epsilon) ** self.alpha

        # 更新优先度
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def get_beta(self) -> float:
        """获取当前 beta 值"""
        return self.beta

    def __len__(self) -> int:
        """获取缓冲区大小"""
        if self.prioritized:
            return self.size
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        if self.prioritized:
            self.priorities = np.zeros(self.capacity)
            self.pos = 0
            self.size = 0
            self.max_priority = 1.0
        else:
            self.buffer.clear()

    def __getitem__(self, idx: int):
        """根据索引获取经验"""
        if self.prioritized:
            return self.buffer[idx]
        return self.buffer[idx]
