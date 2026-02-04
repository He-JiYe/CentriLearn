"""
Metric 模块
提供多种图序列决策任务的评估指标
支持指标注册、组合和自动评估
"""
from centrilearn.metrics.base import BaseMetric
from centrilearn.metrics.manager import MetricManager
from centrilearn.metrics.network_dismantling_metrics import AUC, AttackRate

__all__ = [
    'BaseMetric',
    'MetricManager',
    'AUC',
    'AttackRate'
]



