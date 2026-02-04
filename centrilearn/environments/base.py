"""
网络环境基类
提供网络强化学习环境的标准接口和通用功能
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import networkx as nx
import random
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
    def __init__(self, 
                 graph: nx.Graph = None, 
                 node_features: str = 'ones',
                 synth_type: str = 'ba',
                 synth_args: Dict[str, Any] = None,
                 use_component: bool = False,
                 is_undirected: bool = True, 
                 device: str = 'cpu'):
        """设置原始网络图并重新映射节点编号
        
        Args:
            graph: 网络图对象
            node_features: 节点特征类型 ('ones', 'degree', 'combin')
            synth_type: 合成网络类型 ('ba', 'er', 'ws')
            synth_args: 合成网络参数
            use_component: 是否使用连通分量
            is_undirected: 是否将图转换为无向图
        """
        self.device = device
        self.node_features = node_features
        self.is_synth = graph is None
        self.synth_type = synth_type if self.is_synth else None
        self.synth_args = synth_args if self.is_synth else None
        self.use_component = use_component
        self.is_undirected = is_undirected
        self.reset(graph)

    def reset(self, graph: Optional[nx.Graph] = None) -> Dict[str, Any]:
        """重置环境
        
        Args:
            graph: 网络图对象
            
        Returns:
            初始状态信息
        """
        # Graph > Origin Graph > Synth Graph
        if graph is not None:
            self.graph = nx.freeze(graph)
            self.num_nodes = graph.number_of_nodes()
        
        if self.is_synth:
            min_n, max_n = self.synth_args.pop('min_n', 40), self.synth_args.pop('max_n', 60)
            self.num_nodes = random.randint(min_n, max_n)
            self.synth_args['n'] = self.num_nodes
            if self.synth_type == 'ba':
                self.graph = nx.barabasi_albert_graph(**self.synth_args)
            elif self.synth_type == 'er':
                self.graph = nx.erdos_renyi_graph(**self.synth_args)
            elif self.synth_type == 'ws':
                self.graph = nx.watts_strogatz_graph(**self.synth_args)

        self.edge_index = torch.tensor(list(self.graph.edges()), dtype=torch.long, device=self.device).t().contiguous()
        if self.is_undirected:
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
        if self.use_component:
            component = self.connected_components(edge_index, num_nodes)
        else:
            component = None

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


