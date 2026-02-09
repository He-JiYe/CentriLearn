"""
测试向量化缓冲区和管理器的功能
"""

import torch
import numpy as np
from centrilearn.buffer.replaybuffer import ReplayBuffer, VectorizedReplayBuffer
from centrilearn.buffer.rolloutbuffer import RolloutBuffer, VectorizedRolloutBuffer
from centrilearn.metrics.manager import MetricManager, VectorizedMetricManager
from centrilearn.metrics.base import BaseMetric


class TestMetric(BaseMetric):
    """测试指标类"""
    def __init__(self, name="test_metric"):
        super().__init__(name)
        self.value = 0

    def process(self, state, action, reward, next_state, done, info=None):
        self.value += reward
        return self.value

    def evaluate(self, env=None, model=None, num_episodes=1):
        return {"average": self.value / num_episodes}

    def compute(self) -> float:
        return self.value

    def get_result(self):
        return {"current": self.value}

    def reset(self):
        self.value = 0


def test_vectorized_replaybuffer():
    """测试向量化 ReplayBuffer"""
    print("\n=== 测试向量化 ReplayBuffer ===")
    
    # 创建向量化 ReplayBuffer
    env_num = 4
    buffer = VectorizedReplayBuffer(
        env_num=env_num,
        capacity=1000,
        prioritized=True
    )
    
    print(f"创建了向量化 ReplayBuffer: {buffer}")
    print(f"缓冲区数量: {len(buffer.buffers)}")
    
    # 测试添加经验
    states = [f"state_{i}" for i in range(env_num)]
    actions = [i for i in range(env_num)]
    rewards = [1.0 for _ in range(env_num)]
    next_states = [f"next_state_{i}" for i in range(env_num)]
    dones = [False for _ in range(env_num)]
    
    buffer.push(states, actions, rewards, next_states, dones)
    print("添加经验成功")
    
    # 测试采样
    samples, indices, weights = buffer.sample(batch_size=8)
    print(f"采样成功，获取了 {len(samples)} 个样本")
    
    # 测试更新优先级
    if indices is not None and weights is not None:
        new_priorities = np.ones_like(indices)
        buffer.update_priorities(indices, new_priorities)
        print("更新优先级成功")
    
    # 测试清空缓冲区
    buffer.clear()
    print("清空缓冲区成功")
    print(f"清空后缓冲区大小: {len(buffer)}")
    
    print("向量化 ReplayBuffer 测试通过！")


def test_vectorized_rolloutbuffer():
    """测试向量化 RolloutBuffer"""
    print("\n=== 测试向量化 RolloutBuffer ===")
    
    # 创建向量化 RolloutBuffer
    env_num = 4
    buffer = VectorizedRolloutBuffer(
        env_num=env_num,
        capacity=1000
    )
    
    print(f"创建了向量化 RolloutBuffer: {buffer}")
    print(f"缓冲区数量: {len(buffer.buffers)}")
    
    # 测试添加经验
    states = [f"state_{i}" for i in range(env_num)]
    actions = [torch.tensor(i) for i in range(env_num)]
    log_probs = [torch.tensor(0.5) for _ in range(env_num)]
    rewards = [1.0 for _ in range(env_num)]
    dones = [False for _ in range(env_num)]
    values = [torch.tensor(0.0) for _ in range(env_num)]
    
    buffer.push(states, actions, log_probs, rewards, dones, values)
    print("添加经验成功")
    
    # 测试获取批次
    batches = buffer.get_batches(batch_size=4)
    print(f"获取批次成功，获取了 {len(batches)} 个批次")
    
    # 测试清空缓冲区
    buffer.clear()
    print("清空缓冲区成功")
    print(f"清空后缓冲区大小: {len(buffer)}")
    
    print("向量化 RolloutBuffer 测试通过！")


def test_vectorized_metric_manager():
    """测试向量化 MetricManager"""
    print("\n=== 测试向量化 MetricManager ===")
    
    # 创建向量化 MetricManager
    env_num = 4
    metrics = [TestMetric()]
    manager = VectorizedMetricManager(
        env_num=env_num,
        metrics=metrics
    )
    
    print(f"创建了向量化 MetricManager: {manager}")
    print(f"管理器数量: {len(manager.managers)}")
    
    # 测试更新指标
    states = [f"state_{i}" for i in range(env_num)]
    actions = [i for i in range(env_num)]
    rewards = [1.0 for _ in range(env_num)]
    next_states = [f"next_state_{i}" for i in range(env_num)]
    dones = [False for _ in range(env_num)]
    
    results = manager.update(states, actions, rewards, next_states, dones)
    print(f"更新指标成功，结果: {results}")
    
    # 测试获取结果
    results = manager.get_results()
    print(f"获取结果成功，全局步数: {results['global_step']}")
    
    # 测试获取摘要
    summary = manager.get_summary()
    print(f"获取摘要成功，摘要: {summary}")
    
    # 测试重置指标
    manager.reset()
    print("重置指标成功")
    
    # 测试清空指标
    summary = manager.get_summary()
    print(f"重置后摘要: {summary}")
    
    print("向量化 MetricManager 测试通过！")


if __name__ == "__main__":
    print("开始测试向量化组件...")
    
    # 测试向量化 ReplayBuffer
    test_vectorized_replaybuffer()
    
    # 测试向量化 RolloutBuffer
    test_vectorized_rolloutbuffer()
    
    # 测试向量化 MetricManager
    test_vectorized_metric_manager()
    
    print("\n所有测试通过！向量化组件功能正常。")
