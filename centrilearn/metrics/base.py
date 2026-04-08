"""
Base Metric Class
Defines common interface for all metrics
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMetric(ABC):
    """Base Metric Class

    All custom metrics should inherit from this class and implement relevant methods.

    Args:
        name: Metric name
        record: str ('max', 'min') record max/min historical values
    """

    def __init__(self, name: str = None, record: str = "max"):
        self.name = name if name is not None else self.__class__.__name__
        self.record = record
        self.max_history = None
        self.min_history = None
        self._history: List[float] = []
        self._count = 0
        self._total = 0.0

    @abstractmethod
    def process(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Process single step data.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information

        Returns:
            Metric value for this step (optional)
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """Compute current accumulated metric value.

        Returns:
            Current metric value
        """
        pass

    def update(self, value: float):
        """Update metric accumulated value.

        Args:
            value: Metric value
        """
        self._total += value
        self._count += 1
        self._history.append(value)

        if self.record == "max":
            self.max_history = (
                value if self.max_history is None else max(self.max_history, value)
            )
        elif self.record == "min":
            self.min_history = (
                value if self.min_history is None else min(self.min_history, value)
            )

    def reset(self):
        """Reset metric state."""
        self.max_history = None
        self.min_history = None
        self._count = 0
        self._total = 0.0
        self._history = []

    def get_result(self) -> Dict[str, Any]:
        """Get current metric result.

        Returns:
            Dictionary containing current value, mean, history, etc.
        """
        result = {
            "name": self.name,
            "current": self.compute(),
        }

        if self._count > 0:
            result["mean"] = self._total / self._count
            result["count"] = self._count

        if len(self._history) > 0:
            result["history"] = self._history.copy()

        if self.max_history is not None:
            result["max_history"] = self.max_history

        if self.min_history is not None:
            result["min_history"] = self.min_history

        return result

    @property
    def count(self) -> int:
        """Get count."""
        return self._count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, current={self.compute():.4f})"
        )

    def __call__(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        info: Dict[str, Any] = None,
    ) -> Optional[float]:
        """Convenience call interface."""
        return self.process(state, action, reward, next_state, done, info)
