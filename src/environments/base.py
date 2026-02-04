"""
网络环境基类
提供网络强化学习环境的标准接口和通用功能
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph, degree

class BaseEnv(ABC):
    """网络强化学习环境基类
    
    所有网络相关环境的基类，定义了标准接口和通用功能。
    子类根据需要实现 _reset, step, get_state 逻辑。
    
    Attributes:
        graph: 网络视图 (要求节点标签从 0 - n-1)
        edge_index: 边索引张量 [2, num_edges]
        node_mask: 节点掩码
        num_nodes: 网络节点数
    """
    def __init__(self, graph: nx.Graph, node_features: str = 'ones', is_undirected: bool = True, device: str = 'cpu'):
        """设置原始网络图并重新映射节点编号
        
        Args:
            graph: 网络图对象
            node_features: 节点特征类型 ('ones', 'degree', 'combin')
            is_undirected: 是否将图转换为无向图
        """
        self.device = device
        self.node_features = node_features
        self.reset(graph, is_undirected)

    def reset(self, graph: Optional[nx.Graph] = None, is_undirected: bool = True) -> Dict[str, Any]:
        """重置环境
        
        Args:
            graph: 网络图对象
            is_undirected: 是否将图转换为无向图
            
        Returns:
            初始状态信息
        """
        if graph is not None:
            self.graph = nx.freeze(graph)
            self.num_nodes = graph.number_of_nodes()
            self.edge_index = torch.tensor(list(graph.edges()), dtype=torch.long, device=self.device).t().contiguous()

            if is_undirected:
                self.edge_index = to_undirected(self.edge_index, num_nodes=self.num_nodes)

        self.node_mask = torch.ones(self.num_nodes, dtype=bool, device=self.device)
        
        # 重置剩余统计信息
        self._reset()

        return self.get_state()

    @abstractmethod
    def _reset(self):
        """
        重置剩余统计信息
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """执行一步环境交互
        
        Args:
            action: 执行的动作
            
        Returns:
            reward: 奖励值
            done: 是否终止
            info: 额外信息字典
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        return {
            'mapping': self.node_mask.nonzero(as_tuple=False).view(-1),         # mapping[当前图索引] -> 原始图索引
            'pyg_data': self.get_pyg_data(),
        }

    def get_pyg_data(self, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取PyG格式的剩余图数据
        
        Args:
            mask: 节点掩码

        Returns:
            Data(
            x: 节点特征张量 [num_nodes, feature_dim], 
            edge_index: 边索引张量 [2, num_edges], 
            component: 连通分量标签 [num_nodes]
            )
        """
        node_mask = self.node_mask
        if mask is not None:
            node_mask &= mask

        # 获取剩余图节点索引
        mapping = node_mask.nonzero(as_tuple=False).view(-1)

        edge_index, _ = subgraph(mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
        
        # 构建节点特征
        num_nodes = mapping.shape[0]
        if self.node_features == 'ones':
            x = torch.ones(num_nodes, 2, device=self.device)
        elif self.node_features == 'degree':
            deg = degree(edge_index[0], num_nodes)
            x = torch.stack([deg, deg], dim=1, device=self.device)
        elif self.node_features == 'combin':
            deg = degree(edge_index[0], num_nodes) / num_nodes
            x = torch.cat([torch.ones(num_nodes, 1, device=self.device), deg.unsqueeze(-1)], dim=1)
        else:
            raise ValueError(f"Unknown node_features: {self.node_features}")
        
        # 连通分量标签
        component = self.connected_components(edge_index, num_nodes)
     
        return Data(x=x, edge_index=edge_index, component=component).to(self.device)

    def connected_components(self, edge_index, num_nodes) -> List[int]:
        """
        基于并查集获取剩余图中每个节点的分支标签
        """
        parent = torch.arange(num_nodes, device=self.device)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                if root_x < root_y:
                    parent[root_y] = root_x
                else:
                    parent[root_x] = root_y
        
        # 遍历所有边进行合并
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            union(src, dst)
        
        for i in range(num_nodes):
            find(i)
        
        # 重新映射标签为连续的整数
        component_labels = torch.unique(parent, return_inverse=True)[1]
        
        return component_labels

    def is_empty(self) -> bool:
        """检查图是否为空
        
        Returns:
            是否为空图
        """
        return len(self.graph.nodes()) == 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


