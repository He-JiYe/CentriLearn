"""
网络瓦解专用指标
"""
import numpy as np
from typing import Dict, Any, Optional
from scipy.integrate import trapezoid
from src.metrics.base import BaseMetric
from src.utils.registry import METRICS


@METRICS.register_module()
class AUC(BaseMetric):
    """AUC (Area Under Curve) 指标

    计算一幕游戏中 Attack Curve 的面积, 衡量网络瓦解的效果。
    值越小越好，表示更快地将网络分解为小连通分量。

    Args:
        name: 指标名称
        record: 记录最大/最小历史值
    """

    def __init__(self, name: str = 'AUC', record: str = 'min'):
        super().__init__(name, record)
        self._current_lcc_size = [1]
        self._current_num_nodes = 0

    def process(self,
                _state: Dict[str, Any],
                _action: int,
                _reward: float,
                _next_state: Dict[str, Any],
                done: bool,
                info: Dict[str, Any] = None) -> Optional[float]:
        """处理步骤数据

        从环境的 lcc_size 属性计算 AUC
        """
        self._current_lcc_size.append(info.get('lcc_size'))
        self._current_num_nodes = info.get('num_nodes')

        if done:
            x = np.linspace(0, 1, self._current_num_nodes)
            auc_value = trapezoid(self._current_lcc_size, x[:len(self._current_lcc_size)])

            self.update(auc_value)
            self._current_lcc_size = [1]
            self._current_num_nodes = 0
            return auc_value

        return None

    def evaluate(self,
                 env: Any = None,
                 model: Any = None,
                 num_episodes: int = 1) -> Dict[str, float]:
        """评估 AUC 指标"""
        self.reset()
        self._current_lcc_size = [1]
        self._current_num_nodes = 0

        for _ in range(num_episodes):
            state = env.reset()

            done = False
            while not done:
                if model is not None:
                    action, _ = model.select_action(state)
                else:
                    import random
                    action = random.randint(0, env.num_nodes - 1)

                next_state, reward, done, info = env.step(action, state['mapping'])
                info['lcc_size'], info['num_nodes'] = env.lcc_size[-1], env.num_nodes
                self.process(state, action, reward, next_state, done, info)

                state = next_state

        return self.get_result()

    def compute(self) -> float:
        """返回当前 AUC 平均值"""
        if self._count > 0:
            return self._total / self._count
        return 0.0

    def reset(self):
        """重置指标"""
        super().reset()
        self._current_lcc_size = [1]
        self._current_num_nodes = 0


@METRICS.register_module()
class AttackRate(BaseMetric):
    """攻击率指标 (Attack_Rate)

    计算一幕游戏中行动次数 / 节点数。
    衡量策略在瓦解网络时使用的节点占比。
    值越小越好，表示用更少的节点完成瓦解，效率越高。

    Args:
        name: 指标名称
    """

    def __init__(self, name: str = 'AttackRate', record: str = 'min'):
        super().__init__(name, record)
        self._current_action_count = 0
        self._current_num_nodes = 0

    def process(self,
                _state: Dict[str, Any],
                _action: int,
                _reward: float,
                _next_state: Dict[str, Any],
                done: bool,
                info: Dict[str, Any] = None) -> Optional[float]:
        """处理步骤数据

        从环境的 lcc_size 属性计算 AttackRate
        """
        self._current_action_count += 1
        self._current_num_nodes = info.get('num_nodes')

        if done:
            ar_value = self._current_action_count / self._current_num_nodes

            self.update(ar_value)
            self._current_action_count = 0
            self._current_num_nodes = 0
            return ar_value
        
        return None

    def evaluate(self,
                 env: Any = None,
                 model: Any = None,
                 num_episodes: int = 1) -> Dict[str, float]:
        """评估攻击率指标"""
        self.reset()
        self._current_action_count = 0
        self._current_num_nodes = 0

        for _ in range(num_episodes):
            state = env.reset()

            done = False
            while not done:
                if model is not None:
                    action, _ = model.select_action(state, deterministic=True)
                else:
                    import random
                    action = random.randint(0, env.num_nodes - 1)

                next_state, reward, done, info = env.step(action, state['mapping'])
                info['num_nodes'] = env.num_nodes
                self.process(state, action, reward, next_state, done, info)

                state = next_state

        return self.get_result()

    def compute(self) -> float:
        """返回当前攻击率平均值"""
        if self._count > 0:
            return self._total / self._count
        return 0.0

    def reset(self):
        """重置指标"""
        super().reset()
        self._current_action_count = 0
        self._current_num_nodes = 0
