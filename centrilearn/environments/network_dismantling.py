"""
Network Dismantling Reinforcement Learning Environment
Implements network dismantling task based on node removal
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph, to_scipy_sparse_matrix
from centrilearn.environments.base import BaseEnv
from centrilearn.utils.registry import ENVIRONMENTS


@ENVIRONMENTS.register_module()
class NetworkDismantlingEnv(BaseEnv):
    """Network Dismantling Environment

    Goal: Disrupt network connectivity by removing nodes to minimize the largest connected component.

    Attributes:
        remove_nodes: List of removed nodes
        lcc_size: Largest connected component size history
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        node_features: str = "ones",
        node_dim: int = 2,
        value_type: str = "auc",
        synth_type: str = "ba",
        synth_args: Optional[Dict[str, Any]] = None,
        use_component: bool = False,
        is_undirected: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize network dismantling environment.

        Args:
            graph: Network graph object
            node_features: Node feature type ('ones', 'degree', 'laplacian')
            node_dim: Node feature dimension
            value_type: Reward value type ('auc', 'ar', 'at')
            synth_type: Synthetic network type ('ba', 'er', 'ws')
            synth_args: Synthetic network parameters
            use_component: Whether to use connected components
            is_undirected: Whether to convert graph to undirected
            device: Computing device
            **kwargs: Other parameters
        """
        self.node_features = node_features
        self.node_dim = node_dim
        self.value_type = value_type
        super().__init__(
            graph,
            synth_type=synth_type,
            synth_args=synth_args,
            use_component=use_component,
            is_undirected=is_undirected,
            device=device,
            **kwargs,
        )

    def _reset(self) -> None:
        """Reset environment."""
        self.remove_nodes = []
        self.lcc_size = [1]

    def _step(
        self, action: int, mapping: torch.Tensor = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Execute one step of action.

        Args:
            action: Node index to remove (index in current graph)
            mapping: Node index mapping of state (current graph -> original graph)

        Returns:
            reward: Reward value
            done: Whether episode is done
            info: Additional information dictionary
        """
        if mapping is not None:
            action = mapping[action]

        # Remove node
        self.remove_node(action)

        if self.value_type == "auc":
            reward = -self.lcc() / (self.num_nodes * self.num_nodes)
        elif self.value_type == "ar":
            reward = -1 / self.num_nodes
        elif self.value_type == "at":
            reward = -1
        elif self.value_type == "fiedler":  # fiedler
            reward = self._compute_fiedler(action)


        done = self.is_empty() or self.lcc() <= 2 or self.lcc() / self.num_nodes <= 0.01
        info = {
            "lcc_size": self.lcc_size[-1],
            "num_nodes": self.num_nodes,
        }

        return reward, done, info

    def get_state(
        self, use_gcc: bool = False, mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        node_mask = self.node_mask

        if use_gcc:
            node_mask = torch.logical_and(node_mask, self.lcc_component())
        if mask is not None:
            node_mask = torch.logical_and(node_mask, mask)

        # Get remaining graph node indices
        mapping = node_mask.nonzero(as_tuple=False).view(-1)
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )

        # Construct node features
        num_nodes = mapping.shape[0]
        if self.node_features == "ones":
            x = torch.ones(num_nodes, self.node_dim, device=self.device) 
        elif self.node_features == "degree":
            deg = degree(edge_index[0], num_nodes) / num_nodes
            x = deg.unsqueeze(-1).expand(num_nodes, self.node_dim)
        elif self.node_features == "betweenness":
            G = nx.from_edgelist(edge_index.t().cpu().numpy())
            bet = nx.betweenness_centrality(G, normalized=True)
            bet = torch.tensor([bet.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float, device=self.device) 
            x = bet.unsqueeze(-1).expand(num_nodes, self.node_dim)
        else:
            raise ValueError(f"Unknown node_features: {self.node_features}")

        data = Data(x=x, edge_index=edge_index).to(self.device)

        # Connected component labels
        if self.use_component:
            data.component = self.connected_components(edge_index, num_nodes)

        info = {
            "pyg_data": data,
            "mapping": mapping.view(-1),
            "node_mask": node_mask,
        }

        return info

    def rollout_info(self):
        return {
            "remove_nums": len(self.remove_nodes),
            "remove_nodes": self.remove_nodes,
            "lcc_size": self.lcc_size,
            "num_nodes": self.num_nodes,
        }

    def connected_components(self, edge_index, num_nodes) -> List[int]:
        """Get connected component labels for each node in the remaining graph based on scipy algorithm."""
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        _, component = sp.csgraph.connected_components(adj, directed=False)
        return torch.as_tensor(component, dtype=torch.long, device=edge_index.device)

    def lcc(self) -> int:
        """Return the largest connected component of the remaining graph."""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )
        components = self.connected_components(edge_index, num_nodes)
        components_size = torch.bincount(components)
        return components_size.max().item()

    def lcc_component(self) -> torch.BoolTensor:
        """Return the mask of the largest connected component in the remaining graph."""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )
        components = self.connected_components(edge_index, num_nodes)
        lcc_label = torch.bincount(components).argmax()
        original_indices = mapping[
            (components == lcc_label).nonzero(as_tuple=False).view(-1)
        ]

        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        mask[original_indices] = True
        return mask

    def remove_node(self, node: int):
        """Remove node."""
        self.node_mask[node] = False
        self.remove_nodes.append(int(node))
        self.lcc_size.append(self.lcc() / self.num_nodes)

    def _compute_fiedler(self, action):
        """Compute the fiedler value of the remaining LCC."""
        mapping = self.node_mask.nonzero(as_tuple=False).view(-1)
        num_nodes = mapping.shape[0]
        edge_index, _ = subgraph(
            mapping, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes
        )
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)   # remaining graph
        _, components = sp.csgraph.connected_components(adj, directed=False)
        lcc_label = np.bincount(components).argmax()
        lcc_nodes = np.where(components == lcc_label)[0]
        lcc_adj = adj.tocsr()[lcc_nodes][:, lcc_nodes]
        
        if lcc_adj.shape[0] <= 2:
            return 0.0

        try:
            vals = sp.linalg.eigsh(lcc_adj, k=2, which='SA', return_eigenvectors=False)
            fiedler = vals[1]
        except Exception:
            fiedler = 0.0

        return fiedler * lcc_nodes.shape[0] / self.num_nodes

    def __repr__(self):
        """Print network dismantling environment information."""
        message = (
            super().__repr__()[:-1]
            + f", node_features: {self.node_features}, node_dim: {self.node_dim}, value_type: {self.value_type})"
        )
        return message
