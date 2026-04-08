"""
Basic Graph Neural Network Models
"""

import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer
from tqdm import tqdm


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    fpa (bool, optional): If set to :obj:`True`, enables Feature Pyramid
        Aggregation, which concatenates embeddings from all layers.
        When enabled, the output dimension becomes :obj:`num_layers * hidden_channels`
        (or :obj:`out_channels` if specified).
        Note: :obj:`fpa` and :obj:`jk` are mutually exclusive.
        (default: :obj:`False`)
    graph_aggr (str, optional): The graph pooling method for initializing
        graph embeddings. (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        Graph embeddings are treated as virtual nodes that aggregate
        from all nodes during message passing. (default: :obj:`"add"`)
    **kwargs (optional): Additional arguments of the underlying
        :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        fpa: bool = False,
        graph_aggr: str = "add",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.fpa_mode = fpa
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs
        self.graph_aggr = graph_aggr

        if jk is not None and fpa:
            raise ValueError(
                "Cannot use both 'jk' and 'fpa' modes simultaneously. "
                "Please choose one of them."
            )

        if fpa:
            self.out_channels = (
                out_channels
                if out_channels is not None
                else num_layers * hidden_channels
            )
        elif out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = ModuleList()
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, "forward"):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = "batch" in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(torch.nn.Identity())

        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        # Create linear layer for JK mode or FPA mode with specified out_channels
        if jk is not None:
            if jk == "cat":
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)
        elif fpa and out_channels is not None:
            # FPA mode with specified out_channels
            in_channels = num_layers * hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(
        self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs
    ) -> MessagePassing:
        raise NotImplementedError

    def _pool_graph(self, x: Tensor, batch: OptTensor) -> Tensor:
        """Pool node embeddings to graph embeddings.

        Args:
            x: Node embeddings [num_nodes, hidden_channels]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [num_graphs, hidden_channels]
        """
        if self.graph_aggr == "sum" or self.graph_aggr == "add":
            return global_add_pool(x, batch)
        elif self.graph_aggr == "mean":
            return global_mean_pool(x, batch)
        elif self.graph_aggr == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown graph aggregation: {self.graph_aggr}")

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if hasattr(self, "jk"):
            self.jk.reset_parameters()
        if hasattr(self, "lin"):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (
            num_sampled_nodes_per_hop is not None
            and isinstance(edge_weight, Tensor)
            and isinstance(edge_attr, Tensor)
        ):
            raise NotImplementedError(
                "'trim_to_layer' functionality does not "
                "yet support trimming of both "
                "'edge_weight' and 'edge_attr'"
            )

        xs: List[Tensor] = []
        graph_xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)

        batch_size = batch.max().item() + 1 if batch is not None else 1
        num_nodes = x.size(0)

        # Initialize graph embeddings as virtual nodes
        graph_embed = self._pool_graph(x, batch)
        x_combined = torch.cat([x, graph_embed], dim=0)
        node_indices = torch.arange(num_nodes, device=x.device)
        if batch is not None:
            virtual_node_indices = num_nodes + batch
        else:
            virtual_node_indices = torch.full(
                (num_nodes,), num_nodes, dtype=torch.long, device=x.device
            )

        # Create edges: (node -> virtual_node)
        edges_to_virtual = torch.stack([node_indices, virtual_node_indices], dim=0)

        # Combine original edges and edges to virtual nodes
        edge_index_combined = torch.cat([edge_index, edges_to_virtual], dim=1)

        # Update batch vector for virtual nodes
        if batch is not None:
            virtual_batch = torch.arange(batch_size, device=x.device)
            batch_combined = torch.cat([batch, virtual_batch], dim=0)
        else:
            batch_combined = None

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if not torch.jit.is_scripting() and num_sampled_nodes_per_hop is not None:
                x_combined, edge_index_combined, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x_combined,
                    edge_index_combined,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Message passing on combined graph (nodes + virtual nodes)
            if self.supports_edge_weight and self.supports_edge_attr:
                x_combined = conv(
                    x_combined,
                    edge_index_combined,
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                )
            elif self.supports_edge_weight:
                x_combined = conv(
                    x_combined, edge_index_combined, edge_weight=edge_weight
                )
            elif self.supports_edge_attr:
                x_combined = conv(x_combined, edge_index_combined, edge_attr=edge_attr)
            else:
                x_combined = conv(x_combined, edge_index_combined)

            if i < self.num_layers - 1 or self.jk_mode is not None or self.fpa_mode:
                if self.act is not None and self.act_first:
                    x_combined = self.act(x_combined)
                if self.supports_norm_batch:
                    x_combined = norm(x_combined, batch_combined, batch_size)
                else:
                    x_combined = norm(x_combined)
                if self.act is not None and not self.act_first:
                    x_combined = self.act(x_combined)
                x_combined = self.dropout(x_combined)

                # Feature Pyramid Aggregation mode
                if self.fpa_mode:
                    # Split into node and graph embeddings and save for concatenation
                    x_layer = x_combined[:num_nodes]
                    graph_embed_layer = x_combined[num_nodes:]
                    xs.append(x_layer)
                    graph_xs.append(graph_embed_layer)
                # Jumping Knowledge mode
                elif hasattr(self, "jk"):
                    x = x_combined[:num_nodes]
                    graph_embed = x_combined[num_nodes:]
                    xs.append(x)
                    graph_xs.append(graph_embed)

        # Feature Pyramid Aggregation: concatenate all layer outputs
        if self.fpa_mode:
            x = torch.cat(xs, dim=-1)  # [num_nodes, num_layers * hidden_channels]
            graph_embed = torch.cat(
                graph_xs, dim=-1
            )  # [batch_size, num_layers * hidden_channels]
        # Jumping Knowledge
        elif hasattr(self, "jk"):
            x = x_combined[:num_nodes]
            graph_embed = x_combined[num_nodes:]
            x = self.jk(xs)
            graph_embed = self.jk(graph_xs)
        else:
            x = x_combined[:num_nodes]
            graph_embed = x_combined[num_nodes:]

        x = self.lin(x) if hasattr(self, "lin") else x

        return x, graph_embed
