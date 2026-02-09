"""
网络瓦解强化学习环境
实现基于移除节点的网络瓦解任务
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
import torch
from torch_geometric.utils import subgraph

from centrilearn.utils.registry import ENVIRONMENTS

from .base import BaseEnv


@ENVIRONMENTS.register_module()
class NetworkDismantlingEnv(BaseEnv):
    """网络瓦解环境

    目标：通过移除节点来破坏网络的连通性，使最大连通分量最小化。

    Attributes:
        remove_nodes: 已移除的节点列表
        lcc_size: 最大连通分量大小历史记录
    """

    def __init__(
        self, graph: nx.Graph, value_type: str = "auc", use_gcc: bool = False, **kwargs
    ):
        """初始化网络瓦解环境

        Args:
            graph: 网络图对象
            node_features: 节点特征类型 ('ones', 'degree', 'combin')
            is_undirected: 是否将图转换为无向图
            value_type: ['auc', 'ar']
            use_gcc: 只与最大连通分支进行交互
            device: 设备类型
        """
        self.value_type = value_type
        self.use_gcc = use_gcc
        super().__init__(graph, **kwargs)

    def _reset(self) -> None:
        """重置环境"""
        self.remove_nodes = []
        self.lcc_size = [1]

    def _step_impl(
        self, action: int
        ) -> Tuple[float, bool, Dict[str, Any]]:
        """执行一步动作

        Args:
            action: 要移除的节点索引 (当前图 graph 中的索引)
            mapping: 节点索引映射 (当前图 -> 原始图)

        Returns:
            reward: 奖励值
            done: 是否终止
            info: 额外信息字典
        """

        # 移除节点
        self.remove_node(action)

        next_state = self.get_state()

        if self.value_type == "auc":
            reward = -self.lcc() / (self.num_nodes * self.num_nodes)
        elif self.value_type == "ar":
            reward = -1 / self.num_nodes

        done = self.is_empty() or self.lcc() <= 1
        info = {}

        return next_state, reward, done, info

    def get_state(self) -> Dict[str, Any]:
        if self.use_gcc:
            gcc = self.lcc_component()
            pyg_data = self.get_pyg_data(mask=gcc)
        else:
            pyg_data = self.get_pyg_data()

        info = {
            "mapping": self.node_mask.nonzero(as_tuple=False).view(
                -1
            ),  # mapping[当前图索引] -> 原始图索引
            "pyg_data": pyg_data,
        }

        return info

    def lcc(self) -> int:
        """返回剩余图的最大连通分量"""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )
        components = self.connected_components(edge_index, num_nodes)
        components_size = torch.bincount(components)
        return components_size.max().item()

    def lcc_component(self) -> List:
        """返回剩余图的最大连通分量的索引"""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )
        components = self.connected_components(edge_index, num_nodes)
        lcc = torch.bincount(components).max()[0]
        return components.eq(lcc)

    def remove_node(self, node: int):
        """移除节点 node"""
        masked_node = self.mapping[node]
        self.node_mask[masked_node] = False

        self.remove_nodes.append(masked_node)
        self.lcc_size.append(self.lcc() / self.num_nodes)
