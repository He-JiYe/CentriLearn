"""
Base Environment for Network RL
Provides standard interface and common functionality for network reinforcement learning environments
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import torch
from torch_geometric.utils import to_undirected


class BaseEnv(ABC):
    """Base Environment for Network Reinforcement Learning

    Base class for all network-related environments, defining standard interfaces and common functionality.
    Subclasses implement _reset, step, get_state logic as needed.

    Attributes:
        graph: Network view (requires node labels from 0 to n-1)
        edge_index: Edge index tensor [2, num_edges]
        node_mask: Node mask
        num_nodes: Number of network nodes
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        synth_type: str = "ba",
        synth_args: Optional[Dict[str, Any]] = None,
        use_component: bool = False,
        is_undirected: bool = True,
        device: str = "cpu",
    ):
        """Set up original network graph and remap node indices.

        Args:
            graph: Network graph object
            synth_type: Synthetic network type ('ba', 'er', 'ws')
            synth_args: Synthetic network parameters
            use_component: Whether to use connected components
            is_undirected: Whether to convert graph to undirected
            device: Computing device
        """
        self.device = device
        self.is_synth = graph is None
        self.synth_type = synth_type if self.is_synth else None
        self.synth_args = synth_args if self.is_synth else None
        self.use_component = use_component
        self.is_undirected = is_undirected
        self.reset(graph)

    def reset(self, graph: Optional[nx.Graph] = None, **kwargs) -> Dict[str, Any]:
        """Reset environment.

        Args:
            graph: Network graph object
            **kwargs: Other parameters

        Returns:
            Initial state information
        """
        # Graph > Origin Graph > Synth Graph
        if graph is not None:
            self.graph = nx.freeze(graph)
            self.num_nodes = graph.number_of_nodes()

        if self.is_synth:
            if self.synth_args is None:
                self.synth_args = {"m": 4}

            min_n, max_n = self.synth_args.get("min_n", 40), self.synth_args.get(
                "max_n", 60
            )
            syn_args = {
                k: v for k, v in self.synth_args.items() if k not in ["min_n", "max_n"]
            }
            syn_args["n"] = self.synth_args["n"] = self.num_nodes = random.randint(
                min_n, max_n
            )

            if self.synth_type == "ba":
                self.graph = nx.barabasi_albert_graph(**syn_args)
            elif self.synth_type == "er":
                self.graph = nx.erdos_renyi_graph(**syn_args)
            elif self.synth_type == "ws":
                self.graph = nx.watts_strogatz_graph(**syn_args)

        self.edge_index = (
            torch.tensor(list(self.graph.edges()), dtype=torch.long, device=self.device)
            .t()
            .contiguous()
        )
        if self.is_undirected:
            self.edge_index = to_undirected(self.edge_index, num_nodes=self.num_nodes)

        self.node_mask = torch.ones(self.num_nodes, dtype=bool, device=self.device)
        self.step_count = 0

        # Reset remaining statistics
        self._reset()

        return self.get_state(**kwargs)

    @abstractmethod
    def _reset(self) -> None:
        """Reset remaining statistics."""
        pass

    def step(self, action: int, *args, **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """Execute one step of environment interaction.

        Args:
            action: Action taken

        Returns:
            reward: Reward value
            done: Whether episode is done
            info: Additional information dictionary
        """
        self.step_count += 1
        return self._step(action, *args, **kwargs)

    @abstractmethod
    def _step(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Step logic that subclasses need to implement."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current state information that subclasses need to implement."""
        pass

    @abstractmethod
    def rollout_info(self) -> Dict[str, Any]:
        """Define what information to output after completing a rollout."""
        pass

    def is_empty(self) -> bool:
        """Check if graph is empty.

        Returns:
            Whether the graph is empty
        """
        return self.node_mask.sum() == 0

    def __repr__(self) -> str:
        """Print environment information."""
        return f"{self.__class__.__name__}(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()}, use_component: {self.use_component}, is_undirected: {self.is_undirected}, is_synth: {self.is_synth}, synth_type: {self.synth_type}, synth_args: {self.synth_args}, device: {self.device})"
