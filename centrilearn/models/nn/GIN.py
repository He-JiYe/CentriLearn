"""
The implementation of GIN with graph embedding based on PyG's GINConv.
"""

from typing import Union

import torch
import torch.nn.functional as F
from torch_geometric.nn import (MLP, GINConv, global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import OptTensor

from centrilearn.utils.registry import NN


@NN.register_module()
class GIN(BasicGNN):
    """GIN with graph embedding computation.

    Uses PyG's GINConv for node-level message passing and computes
    graph-level embeddings via global pooling.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        num_layers: Number of GIN layers
        output_dim: Output feature dimension (default: hidden_channels)
        graph_aggr: Graph pooling method ('add', 'mean', 'max')
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        output_dim: int = None,
        aggr: str = "mean",
        graph_aggr: str = "add",
        norm: str = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.graph_aggr = graph_aggr
        self.aggr = aggr

        kwargs.pop("output_dim", None)

        # Initialize BasicGNN with norm option
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            norm=norm,
            **kwargs,
        )

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: OptTensor = None,
        graph_embed: torch.Tensor = None,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass with graph embedding as virtual node.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            graph_embed: Graph embeddings [num_graphs, in_channels] (Default: None)

        Returns:
            Tuple of (node_embeddings, graph_embeddings) [num_graphs, output_dim]
        """
        batch_size = batch.max().item() + 1 if batch is not None else 1

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Update the graph embeddings first
            current_graph_embed = self._pool_graph(x, batch)
            current_graph_embed = conv.lin_l(current_graph_embed)

            if conv.root_weight and graph_embed is not None:
                current_graph_embed = current_graph_embed + conv.lin_r(graph_embed)

            if conv.normalize:
                current_graph_embed = F.normalize(current_graph_embed, p=2.0, dim=-1)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    current_graph_embed = self.act(current_graph_embed)
                if self.supports_norm_batch:
                    current_graph_embed = norm(
                        current_graph_embed,
                        torch.arange(batch_size, device=x.device),
                        batch_size,
                    )
                else:
                    current_graph_embed = norm(current_graph_embed)
                if self.act is not None and not self.act_first:
                    current_graph_embed = self.act(current_graph_embed)
                current_graph_embed = self.dropout(current_graph_embed)

            graph_embed = current_graph_embed

            # Then update the node embeddings
            x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)

        return x, graph_embed
