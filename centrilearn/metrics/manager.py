"""
Metric 管理器
管理多个指标的生命周期、计算和记录
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from centrilearn.metrics.base import BaseMetric


class MetricManager:
    """指标管理器

    负责管理多个指标的生命周期，支持批量和单个更新。

    Args:
        metrics: 指标实例列表或配置字典列表
        save_dir: 保存结果目录（可选）
        log_interval: 日志打印间隔（步数）
    """

    def __init__(
        self,
        metrics: Optional[List[Union[BaseMetric, Dict[str, Any]]]] = None,
        save_dir: Optional[str] = None,
        log_interval: int = 100,
    ):
        self.metrics: List[BaseMetric] = []
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_interval = log_interval
        self._global_step = 0
        self._episode_count = 0
        self._start_time = None

        if metrics:
            self.add_metrics(metrics)

    def add_metric(self, metric: Union[BaseMetric, Dict[str, Any]]):
        """添加单个指标

        Args:
            metric: 指标实例或配置字典
        """
        if isinstance(metric, dict):
            from centrilearn.utils.builder import build_metric

            metric = build_metric(metric)

        if isinstance(metric, BaseMetric):
            from copy import deepcopy

            metric = deepcopy(metric)

        if not isinstance(metric, BaseMetric):
            raise TypeError(f"metric 必须是 BaseMetric 实例或配置字典")

        # 检查名称是否重复
        for existing_metric in self.metrics:
            if existing_metric.name == metric.name:
                raise ValueError(f"指标名称 '{metric.name}' 已存在")

        self.metrics.append(metric)

    def add_metrics(self, metrics: List[Union[BaseMetric, Dict[str, Any]]]):
        """添加多个指标

        Args:
            metrics: 指标实例列表或配置字典列表
        """
        for metric in metrics:
            self.add_metric(metric)

    def remove_metric(self, name: str) -> bool:
        """移除指标

        Args:
            name: 指标名称

        Returns:
            是否成功移除
        """
        for i, metric in enumerate(self.metrics):
            if metric.name == name:
                self.metrics.pop(i)
                return True
        return False

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """获取指标实例

        Args:
            name: 指标名称

        Returns:
            指标实例，如果不存在则返回 None
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """更新所有指标

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
            info: 额外信息

        Returns:
            更新后的指标值字典
        """
        results = {}
        for metric in self.metrics:
            value = metric.process(state, action, reward, next_state, done, info)
            if value is not None:
                results[metric.name] = value

        self._global_step += 1

        if done:
            self._episode_count += 1

        return results

    def evaluate(
        self, env: Any = None, model: Any = None, num_episodes: int = 1
    ) -> Dict[str, Any]:
        """在完整episode上评估所有指标

        Args:
            env: 环境实例
            model: 模型实例
            num_episodes: 评估的episode数量

        Returns:
            所有指标的评估结果
        """
        results = {}
        for metric in self.metrics:
            result = metric.evaluate(env, model, num_episodes)
            results[metric.name] = result

        return results

    def get_results(self) -> Dict[str, Any]:
        """获取所有指标的当前结果

        Returns:
            指标结果字典
        """
        results = {
            "global_step": self._global_step,
            "episode_count": self._episode_count,
            "metrics": {},
        }

        for metric in self.metrics:
            results["metrics"][metric.name] = metric.get_result()

        return results

    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要（仅包含当前值）

        Returns:
            指标值字典
        """
        summary = {}
        for metric in self.metrics:
            result = metric.get_result()
            summary[metric.name] = result.get("current", 0.0)
        return summary

    def reset(self):
        """重置所有指标"""
        for metric in self.metrics:
            metric.reset()
        self._global_step = 0
        self._episode_count = 0

    def reset_metric(self, name: str):
        """重置指定指标

        Args:
            name: 指标名称
        """
        metric = self.get_metric(name)
        if metric:
            metric.reset()

    def save(self, path: Optional[str] = None):
        """保存指标结果

        Args:
            path: 保存路径，默认使用 save_dir
        """
        if path is None:
            if self.save_dir is None:
                raise ValueError("save_dir 未设置，请提供 path 参数")
            path = self.save_dir / f"metrics_step_{self._global_step}.json"

        results = self.get_results()

        # 添加时间信息
        if self._start_time is not None:
            results["elapsed_time"] = time.time() - self._start_time

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def load(self, path: str):
        """加载指标结果（仅用于恢复状态，不支持加载历史记录）

        Args:
            path: 加载路径
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._global_step = data.get("global_step", 0)
        self._episode_count = data.get("episode_count", 0)

    def start_timer(self):
        """开始计时"""
        self._start_time = time.time()

    def get_elapsed_time(self) -> float:
        """获取经过的时间（秒）

        Returns:
            经过的时间
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def log(self, step: Optional[int] = None, prefix: str = ""):
        """打印指标日志

        Args:
            step: 当前步数，默认使用 _global_step
            prefix: 日志前缀
        """
        if step is None:
            step = self._global_step

        print(f"\n{prefix}Step {step} - Episode {self._episode_count}")
        print("-" * 60)

        summary = self.get_summary()
        for name, value in summary.items():
            print(f"  {name:30s}: {value:.4f}")

        if self._start_time is not None:
            print(f"  {'elapsed_time':30s}: {self.get_elapsed_time():.2f}s")

        print("-" * 60)

    def __repr__(self) -> str:
        return f"MetricManager(metrics={len(self.metrics)}, step={self._global_step})"

    def __len__(self) -> int:
        return len(self.metrics)


class VectorizedMetricManager:
    """向量化指标管理器

    管理多个 MetricManager 实例，支持并行操作多个环境的指标。

    Attributes:
        managers: 指标管理器实例列表
        num_envs: 环境数量
        executor: 线程池执行器
    """

    def __init__(
        self,
        env_num: int,
        metrics: Optional[List[Union[BaseMetric, Dict[str, Any]]]] = None,
        save_dir: Optional[str] = None,
        log_interval: int = 100,
    ):
        """初始化向量化指标管理器

        Args:
            env_num: 环境数量
            metrics: 指标实例列表或配置字典列表
            save_dir: 保存结果目录（可选）
            log_interval: 日志打印间隔（步数）
        """
        self.num_envs = env_num
        self.managers = []
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_interval = log_interval

        # 为每个环境创建一个 MetricManager 实例
        for _ in range(env_num):
            self.managers.append(
                MetricManager(
                    metrics=metrics,
                    save_dir=save_dir,
                    log_interval=log_interval,
                )
            )

        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=env_num)

    def add_metric(self, metric: Union[BaseMetric, Dict[str, Any]]):
        """批量添加单个指标到所有管理器

        Args:
            metric: 指标实例或配置字典
        """

        def add_to_manager(manager):
            manager.add_metric(metric)

        list(self.executor.map(add_to_manager, self.managers))

    def add_metrics(self, metrics: List[Union[BaseMetric, Dict[str, Any]]]):
        """批量添加多个指标到所有管理器

        Args:
            metrics: 指标实例列表或配置字典列表
        """
        for metric in metrics:
            self.add_metric(metric)

    def remove_metric(self, name: str) -> bool:
        """批量从所有管理器移除指标

        Args:
            name: 指标名称

        Returns:
            是否所有管理器都成功移除
        """

        def remove_from_manager(manager):
            return manager.remove_metric(name)

        results = list(self.executor.map(remove_from_manager, self.managers))
        return all(results)

    def get_metric(self, name: str, env_idx: int = 0) -> Optional[BaseMetric]:
        """获取指定环境的指标实例

        Args:
            name: 指标名称
            env_idx: 环境索引

        Returns:
            指标实例，如果不存在则返回 None
        """
        if 0 <= env_idx < self.num_envs:
            return self.managers[env_idx].get_metric(name)
        return None

    def update(
        self,
        states: List[Dict[str, Any]],
        actions: List[int],
        rewards: List[float],
        next_states: List[Dict[str, Any]],
        dones: List[bool],
        infos: List[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """批量更新所有管理器的指标

        Args:
            states: 当前状态列表
            actions: 执行的动作列表
            rewards: 获得的奖励列表
            next_states: 下一状态列表
            dones: 是否终止列表
            infos: 额外信息列表

        Returns:
            更新后的指标值字典
        """
        if len(states) != self.num_envs:
            raise ValueError(
                f"states 长度 {len(states)} 必须等于环境数量 {self.num_envs}"
            )

        # 如果 infos 为 None，创建对应长度的 None 列表
        if infos is None:
            infos = [None] * self.num_envs

        # 并行更新每个管理器的指标
        def update_manager(i):
            return self.managers[i].update(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i],
                info=infos[i],
            )

        results = list(self.executor.map(update_manager, range(self.num_envs)))

        # 合并结果
        merged_results = {}
        for i, result in enumerate(results):
            for key, value in result.items():
                merged_results[f"{key}_env{i}"] = value

        return merged_results

    def evaluate(
        self, envs: List[Any] = None, model: Any = None, num_episodes: int = 1
    ) -> Dict[str, Any]:
        """批量评估所有管理器的指标

        Args:
            envs: 环境实例列表
            model: 模型实例
            num_episodes: 评估的episode数量

        Returns:
            所有指标的评估结果
        """

        # 并行评估每个管理器
        def evaluate_manager(i):
            env = envs[i] if envs else None
            return self.managers[i].evaluate(env, model, num_episodes)

        results = list(self.executor.map(evaluate_manager, range(self.num_envs)))

        # 合并结果
        merged_results = {}
        for i, result in enumerate(results):
            merged_results[f"env{i}"] = result

        return merged_results

    def get_results(self) -> Dict[str, Any]:
        """批量获取所有管理器的当前结果

        Returns:
            指标结果字典
        """

        # 并行获取每个管理器的结果
        def get_manager_results(manager):
            return manager.get_results()

        results = list(self.executor.map(get_manager_results, self.managers))

        # 合并结果
        merged_results = {
            "global_step": sum(r["global_step"] for r in results) // self.num_envs,
            "episode_count": sum(r["episode_count"] for r in results) // self.num_envs,
            "metrics": {},
        }

        # 合并每个环境的指标结果
        for i, result in enumerate(results):
            merged_results["metrics"][f"env{i}"] = result["metrics"]

        return merged_results

    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要（仅包含当前值）

        Returns:
            指标值字典
        """

        # 并行获取每个管理器的摘要
        def get_manager_summary(manager):
            return manager.get_summary()

        results = list(self.executor.map(get_manager_summary, self.managers))

        # 合并结果
        merged_summary = {}
        for summary in results:
            for key, value in summary.items():
                merged_summary[f"{key}_avg"] = (
                    merged_summary.get(f"{key}_avg", 0) + value / self.num_envs
                )

        return merged_summary

    def reset(self):
        """批量重置所有管理器的指标"""

        def reset_manager(manager):
            manager.reset()

        list(self.executor.map(reset_manager, self.managers))

    def reset_metric(self, name: str):
        """批量重置所有管理器的指定指标

        Args:
            name: 指标名称
        """

        def reset_manager_metric(manager):
            manager.reset_metric(name)

        list(self.executor.map(reset_manager_metric, self.managers))

    def save(self, path: Optional[str] = None):
        """批量保存所有管理器的指标结果

        Args:
            path: 保存路径，默认使用 save_dir
        """

        def save_manager(i):
            manager_path = path
            if path and self.num_envs > 1:
                # 为每个环境创建不同的保存路径
                import os

                base_dir = os.path.dirname(path)
                base_name = os.path.basename(path)
                name, ext = os.path.splitext(base_name)
                manager_path = os.path.join(base_dir, f"{name}_env{i}{ext}")
            self.managers[i].save(manager_path)

        list(self.executor.map(save_manager, range(self.num_envs)))

    def start_timer(self):
        """批量开始所有管理器的计时"""

        def start_manager_timer(manager):
            manager.start_timer()

        list(self.executor.map(start_manager_timer, self.managers))

    def get_elapsed_time(self) -> float:
        """获取平均经过的时间（秒）

        Returns:
            平均经过的时间
        """

        def get_manager_time(manager):
            return manager.get_elapsed_time()

        times = list(self.executor.map(get_manager_time, self.managers))
        return sum(times) / self.num_envs

    def log(self, step: Optional[int] = None, prefix: str = ""):
        """打印指标日志

        Args:
            step: 当前步数，默认使用平均 global_step
            prefix: 日志前缀
        """
        if step is None:
            step = sum(m._global_step for m in self.managers) // self.num_envs

        episode_count = sum(m._episode_count for m in self.managers) // self.num_envs

        print(f"\n{prefix}Step {step} - Episode {episode_count}")
        print("-" * 60)

        summary = self.get_summary()
        for name, value in summary.items():
            print(f"  {name:30s}: {value:.4f}")

        elapsed_time = self.get_elapsed_time()
        print(f"  {'elapsed_time':30s}: {elapsed_time:.2f}s")

        print("-" * 60)

    def __repr__(self) -> str:
        return f"VectorizedMetricManager(managers={len(self.managers)}, env_num={self.num_envs})"

    def __len__(self) -> int:
        return len(self.managers)

    def __del__(self):
        """清理资源，关闭线程池"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
