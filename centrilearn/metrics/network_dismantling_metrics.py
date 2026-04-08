"""
Network Dismantling Specific Metrics
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import trapezoid

from centrilearn.metrics.base import BaseMetric
from centrilearn.utils.registry import METRICS


@METRICS.register_module()
class AUC(BaseMetric):
    """AUC (Area Under Curve) Metric

    Computes the area under the Attack Curve in one episode, measuring network dismantling effectiveness.
    Smaller is better, indicating faster decomposition of the network into small connected components.

    Args:
        name: Metric name
        record: Record max/min historical values
    """

    def __init__(self, name: str = "AUC", record: str = "min"):
        super().__init__(name, record)
        self._current_lcc_size = [1]
        self._current_num_nodes = 0

    def process(
        self,
        _state: Dict[str, Any],
        _action: int,
        _next_state: Dict[str, Any],
        _reward: float,
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Process step data.

        Compute AUC from the environment's lcc_size attribute.
        """
        self._current_lcc_size.append(info.get("lcc_size"))
        self._current_num_nodes = info.get("num_nodes")

        if done:
            n = self._current_num_nodes
            x = np.linspace(1.0 / n, 1, n)
            auc_value = trapezoid(
                self._current_lcc_size, x[: len(self._current_lcc_size)]
            )

            self.update(auc_value)
            self._current_lcc_size = [1]
            self._current_num_nodes = 0
            return auc_value

        return None

    def compute(self) -> float:
        """Return current AUC average."""
        if self._count > 0:
            return self._history[-1]
        return 0.0

    def reset(self):
        """Reset metric."""
        super().reset()
        self._current_lcc_size = [1]
        self._current_num_nodes = 0


@METRICS.register_module()
class AttackRate(BaseMetric):
    """Attack Rate Metric (Attack_Rate)

    Computes number of actions / number of nodes in one episode.
    Measures the proportion of nodes used by the strategy to dismantle the network.
    Smaller is better, indicating higher efficiency with fewer nodes to complete dismantling.

    Args:
        name: Metric name
    """

    def __init__(self, name: str = "AttackRate", record: str = "min"):
        super().__init__(name, record)
        self._current_action_count = 0
        self._current_num_nodes = 0

    def process(
        self,
        _state: Dict[str, Any],
        _action: int,
        _next_state: Dict[str, Any],
        _reward: float,
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Process step data.

        Compute AttackRate from the environment's lcc_size attribute.
        """
        self._current_action_count += 1
        self._current_num_nodes = info.get("num_nodes")

        if done:
            ar_value = self._current_action_count / self._current_num_nodes

            self.update(ar_value)
            self._current_action_count = 0
            self._current_num_nodes = 0
            return ar_value

        return None

    def compute(self) -> float:
        """Return current attack rate average."""
        if self._count > 0:
            return self._history[-1]
        return 0.0

    def reset(self):
        """Reset metric."""
        super().reset()
        self._current_action_count = 0
        self._current_num_nodes = 0
