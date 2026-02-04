"""
Metric 管理器
管理多个指标的生命周期、计算和记录
"""
from typing import Dict, Any, List, Optional, Union
import time
import json
from pathlib import Path
from centrilearn.metrics.base import BaseMetric


class MetricManager:
    """指标管理器

    负责管理多个指标的生命周期，支持批量和单个更新。

    Args:
        metrics: 指标实例列表或配置字典列表
        save_dir: 保存结果目录（可选）
        log_interval: 日志打印间隔（步数）
    """

    def __init__(self, 
                 metrics: Optional[List[Union[BaseMetric, Dict[str, Any]]]] = None,
                 save_dir: Optional[str] = None,
                 log_interval: int = 100):
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

    def update(self,
               state: Dict[str, Any],
               action: int,
               reward: float,
               next_state: Dict[str, Any],
               done: bool,
               info: Dict[str, Any] = None) -> Dict[str, float]:
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

    def evaluate(self,
                 env: Any = None,
                 model: Any = None,
                 num_episodes: int = 1) -> Dict[str, Any]:
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
            'global_step': self._global_step,
            'episode_count': self._episode_count,
            'metrics': {}
        }

        for metric in self.metrics:
            results['metrics'][metric.name] = metric.get_result()

        return results

    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要（仅包含当前值）

        Returns:
            指标值字典
        """
        summary = {}
        for metric in self.metrics:
            result = metric.get_result()
            summary[metric.name] = result.get('current', 0.0)
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
            results['elapsed_time'] = time.time() - self._start_time

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def load(self, path: str):
        """加载指标结果（仅用于恢复状态，不支持加载历史记录）

        Args:
            path: 加载路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._global_step = data.get('global_step', 0)
        self._episode_count = data.get('episode_count', 0)

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
