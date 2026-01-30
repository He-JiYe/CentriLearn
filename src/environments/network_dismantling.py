"""
网络瓦解强化学习环境
实现基于移除节点的网络瓦解任务
"""
import networkx as nx
import torch
from typing import List, Tuple, Dict, Any
from .base import BaseEnv
from torch_geometric.utils import subgraph
from src.utils.registry import ENVIRONMENTS


@ENVIRONMENTS.register_module()
class NetworkDismantlingEnv(BaseEnv):
    """网络瓦解环境

    目标：通过移除节点来破坏网络的连通性，使最大连通分量最小化。

    Attributes:
        remove_nodes: 已移除的节点列表
        lcc_size: 最大连通分量大小历史记录
    """

    def __init__(self, graph: nx.Graph, value_type: str = 'auc', is_undirected: bool = True, device: str = 'cpu'):
        """初始化网络瓦解环境

        Args:
            graph: 网络图对象
            value_type: ['auc', 'ar']
        """
        super().__init__(graph, is_undirected=is_undirected, device=device)
        self.value_type = value_type
    
    def _reset(self) -> None:
        """重置环境"""
        self.remove_nodes = []
        self.lcc_size = [1]

    def step(self, action: int, mapping: Dict[int, int]) -> Tuple[float, bool, Dict[str, Any]]:
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
        self.remove_node(action, mapping)

        if self.value_type == 'auc':
            reward = -self.lcc() / (self.num_nodes * self.num_nodes)
        elif self.value_type == 'ar':
            reward = -1 / self.num_nodes

        done = self.is_empty() or self.lcc() <= 1
        info = {}
        
        return reward, done, info
    
    def get_state(self) -> Dict[str, Any]:
        info = super().get_state()
        return info

    def lcc(self) -> int:
        """返回剩余图的最大连通分量"""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
        components = self.connected_components(edge_index, num_nodes)
        components_size = torch.bincount(components)
        return components_size.max().item()

    def lcc_component(self) -> List:
        """返回剩余图的最大连通分量的索引"""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
        components = self.connected_components(edge_index, num_nodes)
        components_size = torch.bincount(components)
        return components_size.max()[1]

    def remove_node(self, node: int, mapping: Dict[int, int]):
        """移除节点 node"""
        masked_node = mapping[node]
        self.node_mask[masked_node] = False

if __name__ == "__main__":
    graph = nx.barabasi_albert_graph(50, 2)
    env = NetworkDismantlingEnv(graph)

    print(env)
    print(env.is_empty())
    env.node_mask[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]] = False
    data = env.get_pyg_data()
    print(data.component)