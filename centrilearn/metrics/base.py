"""
Metric 基类
定义所有指标的通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMetric(ABC):
    """指标基类

    所有自定义指标都应该继承此类并实现相关方法。

    Args:
        name: 指标名称
        record: str ('max', 'min') 记录最大/最小历史值
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
        """处理单个步骤的数据

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
            info: 额外信息

        Returns:
            该步骤的指标值（可选）
        """
        pass

    @abstractmethod
    def evaluate(
        self, env: Any = None, model: Any = None, num_episodes: int = 1
    ) -> Dict[str, float]:
        """在完整episode上评估指标

        Args:
            env: 环境实例
            model: 模型实例
            num_episodes: 评估的episode数量

        Returns:
            指标评估结果字典
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """计算当前累积的指标值

        Returns:
            当前指标值
        """
        pass

    def update(self, value: float):
        """更新指标累积值

        Args:
            value: 指标值
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
        """重置指标状态"""
        self.max_history = None
        self.min_history = None
        self._count = 0
        self._total = 0.0
        self._history = []

    def get_result(self) -> Dict[str, Any]:
        """获取当前指标结果

        Returns:
            包含当前值、平均值、历史记录等的字典
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
        """获取计数"""
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
        """便捷调用接口"""
        return self.process(state, action, reward, next_state, done, info)
