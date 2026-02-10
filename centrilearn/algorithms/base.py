"""
强化学习算法基类
定义算法的标准接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from centrilearn.environments.vectorized_env import VectorizedEnv
from centrilearn.utils import (build_metric_manager, build_optimizer,
                               build_replaybuffer, build_scheduler)


class BaseAlgorithm(ABC):
    """强化学习算法基类

    定义算法的通用接口，包括训练、评估、保存/加载等功能。

    Attributes:
        model: 模型实例或模型参数
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        replaybuffer: 经验回放缓冲区
        metric_manager: 指标管理器（可选）
        device: 运行设备
        training_step: 当前训练步数
        model_cfg: 模型配置
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict[str, Any]],
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        replaybuffer_cfg: Optional[Dict[str, Any]] = None,
        metric_manager_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """初始化算法

        Args:
            model: 神经网络模型实例 或 模型配置字典
                   如果是字典，将自动构建模型
            optimizer_cfg: 优化器配置，例如 {'type': 'Adam', 'lr': 1e-4}
            scheduler_cfg: 学习率调度器配置，例如 {'type': 'StepLR', 'step_size': 100}
            replaybuffer_cfg: 经验回放缓冲区配置
            metric_manager: 指标管理器配置（可选）
            device: 运行设备
        """
        self.device = device

        if isinstance(model, nn.Module):
            self.model = model.to(device)
            self.model_cfg = None
        elif isinstance(model, dict):
            self.model_cfg = model
            self.model = self._build_model(model).to(device)
        else:
            raise TypeError(f"model 必须是 nn.Module 或 Dict，当前类型: {type(model)}")

        # 构建优化器、调度器和指标管理器
        self.optimizer = build_optimizer(self.model, optimizer_cfg)
        self.scheduler = build_scheduler(self.optimizer, scheduler_cfg)
        self.replay_buffer = (
            build_replaybuffer(replaybuffer_cfg) if replaybuffer_cfg else None
        )
        self.metric_manager = build_metric_manager(metric_manager_cfg)

        # 训练状态
        self.training_step = 0

    @abstractmethod
    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """从配置构建模型

        子类应该重写此方法以定义特定的模型构建逻辑。

        Args:
            model_cfg: 模型配置字典

        Returns:
            构建好的模型实例
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一步训练

        Args:
            batch: 训练数据批次

        Returns:
            训练指标字典
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """更新模型参数

        Returns:
            更新指标字典
        """
        pass

    @abstractmethod
    def _select_action_single(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """为单个环境选择动作

        Args:
            state: 当前状态
            **kwargs: 算法特定的参数（如 epsilon, deterministic）

        Returns:
            选择动作的相关信息（动作本身以及可能的额外信息）
        """
        pass

    @abstractmethod
    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """收集经验到缓冲区

        Args:
            state: 当前状态
            *args: 其他必需参数（如 action, reward, next_state, done, log_prob, value 等）
            **kwargs: 可选参数
        """
        pass

    def select_action(
        self, state: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs
    ) -> Tuple[Union[torch.Tensor, int, List[Union[torch.Tensor, int]]], ...]:
        """选择动作

        Args:
            state: 当前状态或状态列表（对于向量化环境）
            **kwargs: 算法特定的参数（如 epsilon, deterministic）

        Returns:
            选择动作的相关信息（动作本身以及可能的额外信息）
        """
        # 检查是否为向量化环境的状态列表
        if isinstance(state, list):
            # 获取第一个状态的动作和额外信息，确定返回值结构
            first_result = self._select_action_single(state[0], **kwargs)
            num_return_values = len(first_result)

            # 向量化环境：遍历处理每个状态
            return_values = [[first_result[i]] for i in range(num_return_values)]

            # 处理所有状态
            for s in state[1:]:
                result = self._select_action_single(s, **kwargs)
                for j, value in enumerate(result):
                    return_values[j].append(value)

            return tuple(return_values)
        else:
            # 单环境：直接调用 _select_action_single
            return self._select_action_single(state, **kwargs)

    def set_train_mode(self) -> None:
        """设置为训练模式"""
        self.model.train()

    def set_eval_mode(self) -> None:
        """设置为评估模式"""
        self.model.eval()

    def save_checkpoint(self, path: str, **kwargs):
        """保存检查点

        Args:
            path: 保存路径
            **kwargs: 额外保存的信息
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            **kwargs,
        }

        # 保存调度器状态（如果存在）
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点

        Args:
            path: 检查点路径

        Returns:
            检查点字典
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        # 恢复调度器状态（如果存在）
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def step_scheduler(self, metrics: Optional[Dict[str, float]] = None):
        """更新学习率调度器

        Args:
            metrics: 指标字典（某些调度器如 ReduceLROnPlateau 需要）
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    metric_value = next(iter(metrics.values()))
                    self.scheduler.step(metric_value)
            else:
                self.scheduler.step()

    def get_lr(self) -> float:
        """获取当前学习率

        Returns:
            当前学习率
        """
        return self.optimizer.param_groups[0]["lr"]

    def get_model(self) -> nn.Module:
        """获取模型"""
        return self.model

    def _run_training_loop(
        self, env: Any, training_cfg: Dict[str, Any], verbose: bool = True
    ) -> Dict[str, Any]:
        """通用训练循环实现

        Args:
            env: 环境实例
            training_cfg: 训练配置
            verbose: 是否打印日志

        Returns:
            训练结果字典
        """
        # 获取训练参数
        batch_size = training_cfg.get("batch_size", 32)
        max_steps = training_cfg.get("max_steps", 1000)
        num_episodes = training_cfg.get("num_episodes", 1000)
        log_interval = training_cfg.get("log_interval", 10)
        is_eval = training_cfg.get("is_eval", False)
        eval_interval = training_cfg.get("eval_interval", 50)
        eval_episodes = training_cfg.get("eval_episodes", 5)
        save_interval = training_cfg.get("save_interval", 100)
        save_path = training_cfg.get("save_path", "checkpoints")

        # 初始化
        total_reward = 0
        episode_rewards = []
        completed_episodes = 0  # 已完成的幕数

        if verbose:
            print("\n" + "=" * 60)
            print(f"开始 {self.__class__.__name__} 训练...")
            print("=" * 60)

        if self.metric_manager is not None:
            self.metric_manager.start_timer()

        # 检查是否为向量化环境
        is_vectorized = isinstance(env, VectorizedEnv)

        # 初始化状态
        state = env.reset()
        Returns = 0.0 if not is_vectorized else [0.0 for _ in range(env.num_envs)]

        # 训练循环：直到完成指定的幕数
        while completed_episodes < num_episodes:
            # 选择动作
            action_info = self.select_action(state)
            action = action_info[0]
            extra_info = action_info[1:]

            # 执行动作
            if is_vectorized:
                # 向量化环境不需要 mapping 参数
                next_state, reward, done, info = env.step(action)
                # 更新奖励
                Returns = [Returns[i] + reward[i] for i in range(env.num_envs)]
                # 计算总奖励
                total_reward += sum(reward)
                # 收集经验
                self.collect_experience(
                    state, action, reward, next_state, done, *extra_info
                )

                # 处理指标更新
                if self.metric_manager is not None:
                    self.metric_manager.update(
                        state, action, reward, next_state, done, info
                    )

                # 更新模型
                if self.replay_buffer and len(self.replay_buffer) >= batch_size:
                    metrics = self.update(batch_size)
                    self.step_scheduler(metrics)

                # 检查是否有环境完成
                done_indices = [
                    i
                    for i, ev in enumerate(env)
                    if ev.step_count >= max_steps or done[i]
                ]
                if done_indices:
                    # 增加已完成的幕数
                    completed_episodes += len(done_indices)
                    # 重置已完成的环境
                    reset_states = env.reset(done_indices)
                    # 更新状态
                    for i, idx in enumerate(done_indices):
                        next_state[idx] = reset_states[i]
                        # 记录奖励
                        episode_rewards.append(Returns[idx])
                        Returns[idx] = 0.0

                    # 更新状态
                    state = next_state
                else:
                    state = next_state
                    continue
            else:
                # 单环境
                next_state, reward, done, info = env.step(action)
                Returns += reward
                total_reward += reward
                # 收集经验
                self.collect_experience(
                    state, action, reward, next_state, done, *extra_info
                )

                if self.metric_manager is not None:
                    self.metric_manager.update(
                        state, action, reward, next_state, done, info
                    )

                # 更新模型
                if self.replay_buffer and len(self.replay_buffer) >= batch_size:
                    metrics = self.update(batch_size)
                    self.step_scheduler(metrics)

                if done or env.step_count >= max_steps:
                    completed_episodes += 1
                    episode_rewards.append(Returns)
                    Returns = 0.0
                    state = env.reset()
                else:
                    state = next_state
                    continue

            # 打印训练日志
            if (
                verbose
                and completed_episodes > 0
                and completed_episodes % log_interval == 0
            ):
                avg_reward = sum(episode_rewards[-log_interval:]) / len(
                    episode_rewards[-log_interval:]
                )
                print(
                    f"Episode {completed_episodes:4d} | Reward: {episode_rewards[-1]:8.4f} | "
                    f"Avg Reward ({log_interval}): {avg_reward:8.4f} | LR: {self.get_lr():.6f}"
                )

                # 打印指标信息
                if self.metric_manager is not None:
                    self.metric_manager.log(prefix="  ")

            # 定期评估
            if (
                is_eval
                and completed_episodes > 0
                and completed_episodes % eval_interval == 0
                and self.metric_manager is not None
            ):
                if verbose:
                    print(f"\n  [评估 Episode {completed_episodes}]")
                self.set_eval_mode()
                eval_results = self.metric_manager.evaluate(env, self, eval_episodes)
                if verbose:
                    for name, result in eval_results.items():
                        current = result.get("current", 0.0)
                        print(f"    {name}: {current:.4f}")
                self.set_train_mode()

            # 定期保存模型参数
            if completed_episodes > 0 and completed_episodes % save_interval == 0:
                import os

                os.makedirs(save_path, exist_ok=True)
                checkpoint_path = os.path.join(
                    save_path, f"checkpoint_episode_{completed_episodes}.pth"
                )
                self.save_checkpoint(checkpoint_path, episode=completed_episodes)
                if verbose:
                    print(f"  [保存模型参数] 检查点已保存到: {checkpoint_path}")

        return {
            "total_episodes": num_episodes,
            "total_reward": total_reward,
            "avg_reward": total_reward / num_episodes,
            "final_lr": self.get_lr(),
            "metrics": (
                self.metric_manager.get_results() if self.metric_manager else None
            ),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, device={self.device})"
