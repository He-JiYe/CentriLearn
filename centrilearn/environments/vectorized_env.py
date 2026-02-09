"""
向量化环境基类
支持并行运行多个环境实例，提高采样效率
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

from .base import BaseEnv


class VectorizedEnv:
    """向量化环境包装器

    并行运行多个环境实例，提供批量交互接口。
    支持所有继承自 BaseEnv 的环境类。

    Attributes:
        envs: 环境实例列表
        num_envs: 环境数量
        observations: 当前观测列表
        infos: 环境信息列表
    """

    def __init__(self, env_class: Type[BaseEnv], env_kwargs_list: List[Dict[str, Any]], env_num: Optional[int] = None):
        """初始化向量化环境

        Args:
            env_class: 环境类（必须继承自 BaseEnv）
            env_kwargs_list: 每个环境的初始化参数列表
            env_num: 环境数量，如果传入且 env_kwargs_list 只有一个元素，则创建 env_num 个实例
        """
        if not issubclass(env_class, BaseEnv):
            raise TypeError(f"env_class 必须继承自 BaseEnv，当前为 {env_class}")

        if len(env_kwargs_list) == 0:
            raise ValueError("env_kwargs_list 不能为空")

        # 如果传入了 env_num 且只有一个环境参数，则创建多个实例
        if env_num is not None and len(env_kwargs_list) == 1:
            self.env_kwargs_list = env_kwargs_list * env_num
            self.num_envs = env_num
        else:
            self.env_kwargs_list = env_kwargs_list
            self.num_envs = len(env_kwargs_list)

        self.env_class = env_class

        # 创建环境实例
        self.envs = [env_class(**kwargs) for kwargs in self.env_kwargs_list]

        # 跟踪每个环境的状态
        self.observations: List[Any] = None
        self.dones = [False] * self.num_envs
        self.infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs)

    @classmethod
    def from_graph_list(
        cls,
        env_class: Type[BaseEnv],
        graph_list: List,
        common_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "VectorizedEnv":
        """从图列表创建向量化环境（适用于网络环境）

        Args:
            env_class: 环境类
            graph_list: 网络图列表
            common_kwargs: 共享的环境参数

        Returns:
            VectorizedEnv 实例
        """
        common_kwargs = common_kwargs or {}
        env_kwargs_list = [{"graph": graph, **common_kwargs} for graph in graph_list]
        return cls(env_class, env_kwargs_list)

    def reset(self, indices: Optional[List[int]] = None) -> List[Any]:
        """重置环境

        Args:
            indices: 要重置的环境索引列表，None 表示重置所有环境

        Returns:
            重置后的观测列表
        """
        if self.observations is None:
            # 首次重置所有环境
            def reset_env(i):
                return self.envs[i].reset()
            
            self.observations = list(self.executor.map(reset_env, range(self.num_envs)))
            self.dones = [False] * self.num_envs
            self.infos = [{} for _ in range(self.num_envs)]
            return self.observations

        if indices is None:
            indices = list(range(self.num_envs))

        # 并行重置指定的环境
        def reset_env(idx):
            obs = self.envs[idx].reset()
            self.observations[idx] = obs
            self.dones[idx] = False
            self.infos[idx] = {}
            return obs
        
        results = list(self.executor.map(reset_env, indices))

        return results

    def step(
        self, actions: List[int]
    ) -> Tuple[List[Any], List[float], List[bool], List[Dict[str, Any]]]:
        """执行批量环境步进

        Args:
            actions: 每个环境的动作列表

        Returns:
            observations: 新观测列表
            rewards: 奖励列表
            dones: 终止标志列表
            infos: 信息字典列表
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions 长度 {len(actions)} 必须等于环境数量 {self.num_envs}"
            )

        # 定义每个环境的 step 函数
        def step_env(i):
            env = self.envs[i]
            action = actions[i]
            
            if self.dones[i]:
                # 已终止的环境返回观测但不执行动作
                obs = self.observations[i]
                reward = 0.0
                done = True
                info = {"terminated": True}
            else:
                # 提取 mapping 并执行 step
                state = self.observations[i]
                mapping = env.mapping
                step_result = env.step(action)
                
                # 处理不同的 step 返回值格式
                if len(step_result) == 4:
                    # 格式: (next_state, reward, done, info)
                    obs, reward, done, info = step_result
                else:
                    # 格式: (reward, done, info)
                    reward, done, info = step_result
                    obs = env.get_state()
                
                self.observations[i] = obs
            
            self.dones[i] = done
            return obs, reward, done, info

        # 并行执行所有环境的 step 操作
        results = list(self.executor.map(step_env, range(self.num_envs)))

        # 分离结果
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def __len__(self) -> int:
        """返回环境数量"""
        return self.num_envs

    def __getitem__(self, index: int) -> BaseEnv:
        """获取单个环境实例"""
        return self.envs[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env_class={self.env_class.__name__}, num_envs={self.num_envs})"

    def __del__(self):
        """清理资源，关闭线程池"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
