"""
Metric Manager
Manages lifecycle, computation and recording of multiple metrics
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from centrilearn.metrics.base import BaseMetric


class MetricManager:
    """Metric Manager

    Responsible for managing lifecycle of multiple metrics, supports batch and single updates.

    Args:
        metrics: List of metric instances or configuration dictionaries
    """

    def __init__(
        self,
        metrics: Optional[List[Union[BaseMetric, Dict[str, Any]]]] = None,
    ):
        self.metrics: List[BaseMetric] = []
        self._global_step = 0
        self._episode_count = 0
        self._start_time = None

        if metrics:
            self.add_metrics(metrics)

    def add_metric(self, metric: Union[BaseMetric, Dict[str, Any]]):
        """Add a single metric.

        Args:
            metric: Metric instance or configuration dictionary
        """
        if isinstance(metric, dict):
            from centrilearn.utils.builder import build_metric

            metric = build_metric(metric)

        if not isinstance(metric, BaseMetric):
            raise TypeError(
                f"metric must be a BaseMetric instance or configuration dictionary"
            )

        # Check for duplicate names
        for existing_metric in self.metrics:
            if existing_metric.name == metric.name:
                raise ValueError(f"Metric name '{metric.name}' already exists")

        self.metrics.append(metric)

    def add_metrics(self, metrics: List[Union[BaseMetric, Dict[str, Any]]]):
        """Add multiple metrics.

        Args:
            metrics: List of metric instances or configuration dictionaries
        """
        for metric in metrics:
            self.add_metric(metric)

    def remove_metric(self, name: str) -> bool:
        """Remove metric.

        Args:
            name: Metric name

        Returns:
            Whether successfully removed
        """
        for i, metric in enumerate(self.metrics):
            if metric.name == name:
                self.metrics.pop(i)
                return True
        return False

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get metric instance.

        Args:
            name: Metric name

        Returns:
            Metric instance, or None if not found
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def update(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Update all metrics.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
            info: Additional information

        Returns:
            Updated metrics dictionary
        """
        results = {}
        for metric in self.metrics:
            value = metric.process(state, action, next_state, reward, done, info)
            if value is not None:
                results[metric.name] = value

        self._global_step += 1

        if done:
            self._episode_count += 1

        return results

    def get_results(self) -> Dict[str, Any]:
        """Get current results for all metrics.

        Returns:
            Metrics results dictionary
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
        """Get metrics summary (only current values).

        Returns:
            Metrics values dictionary
        """
        summary = {}
        for metric in self.metrics:
            result = metric.get_result()
            summary[metric.name] = result.get("current", 0.0)
        return summary

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()
        self._global_step = 0
        self._episode_count = 0

    def reset_metric(self, name: str):
        """Reset specified metric.

        Args:
            name: Metric name
        """
        metric = self.get_metric(name)
        if metric:
            metric.reset()

    def start_timer(self):
        """Start timer."""
        self._start_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def __repr__(self) -> str:
        return f"MetricManager(metrics={len(self.metrics)}, step={self._global_step})"

    def __len__(self) -> int:
        return len(self.metrics)
