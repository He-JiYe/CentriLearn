"""强化学习环境模块"""

from .base import BaseEnv
from .network_dismantling import NetworkDismantlingEnv
from .vectorized_env import VectorizedEnv

# 在模块导入后注册环境（延迟导入注册器）
def _register_environments():
    from src.utils.registry import ENVIRONMENTS

    # 注册 NetworkDismantlingEnv（使用 force=True 覆盖）
    ENVIRONMENTS.register_module(name='NetworkDismantlingEnv', force=True)(NetworkDismantlingEnv)

_register_environments()

__all__ = ['BaseEnv', 'NetworkDismantlingEnv', 'VectorizedEnv']
