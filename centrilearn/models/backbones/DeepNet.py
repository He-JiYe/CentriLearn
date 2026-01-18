"""
Deep network backbone with residual connections for graph reinforcement learning.
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torch_geometric.typing import OptTensor

from centrilearn.utils.builder import build_nn
from centrilearn.utils.registry import BACKBONES


class Block(nn.Module):
    """A single nn block with optional residual connection.

    Similar to ResNet block, this block wraps a single-layer nn
    and applies normalization, activation, and optional residual connection.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        aggr: nn aggregation method.
        graph_aggr: Graph pooling method.
        norm: Normalization type.
        dropout: Dropout probability.
        use_residual: Whether to use residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nn: str = "GraphSAGE",
        aggr: str = "mean",
        graph_aggr: str = "add",
        norm: str = "layer",
        dropout: float = 0.0,
        use_residual: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.channel_match = in_channels == out_channels

        # Single-layer GraphSAGE
        self.conv = build_nn(
            {
                "type": nn,
                "in_channels": in_channels,
                "hidden_channels": out_channels,
                "num_layers": 1,
                "output_dim": out_channels,
                "aggr": aggr,
                "graph_aggr": graph_aggr,
                "norm": norm,
                "dropout": dropout,
            }
        )

        # Normalization
        if norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        self.act = nn.ReLU(inplace=True)

        # Projection for residual if dimensions don't match
        if use_residual and not self.channel_match:
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: OptTensor = None,
        graph_embed: torch.Tensor = None,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            graph_embed: Graph embedding [num_graphs, input_dim] (Default: None)

        Returns:
            Updated node features and graph embedding [num_nodes, output_dim]
        """
        node_identity, graph_identity = x, graph_embed

        # Graph convolution
        node_out, graph_out = self.conv(x, edge_index, batch, graph_embed)

        # Residual connection
        if self.use_residual:
            if not self.channel_match:
                node_identity = self.proj(node_identity)
                if graph_identity is not None:
                    graph_identity = self.proj(graph_identity)
            node_out = node_out + node_identity
            if graph_identity is not None:
                graph_out = graph_out + graph_identity

        # Normalization and activation
        node_out = self.norm(node_out)
        graph_out = self.norm(graph_out)
        node_out = self.act(node_out)
        graph_out = self.act(graph_out)

        return node_out, graph_out


@BACKBONES.register_module()
class DeepNet(nn.Module):
    """Deep network backbone with residual connections (ResNet-style).

    This backbone stacks multiple GraphSAGE blocks with residual connections.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_blocks: Number of GraphSAGE blocks.
        block_config: Configuration for each block (dict or list of dicts).
            If single dict, all blocks share same configuration.
            If list of dicts, each block has its own configuration.
        aggr: GraphSAGE aggregation method ('mean', 'max', 'sum').
        graph_aggr: Graph pooling method ('add', 'mean', 'max').
        norm: Normalization type ('layer', 'batch', or None).
        dropout: Dropout probability.
        use_residual: Whether to use residual connections.
        output_dim: Output feature dimension (default: hidden_dim).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        block_config: dict = None,
        aggr: str = "mean",
        graph_aggr: str = "add",
        norm: str = "layer",
        dropout: float = 0.0,
        use_residual: bool = True,
        output_dim: int = None,
        nn: str = "GraphSAGE",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._output_dim = output_dim if output_dim else hidden_channels
        self.use_residual = use_residual

        # Node to embedding
        self.fc = nn.Linear(in_channels, hidden_channels)

        # Build blocks
        self.blocks = self._make_blocks(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            block_config=block_config,
            aggr=aggr,
            graph_aggr=graph_aggr,
            norm=norm,
            dropout=dropout,
            use_residual=use_residual,
        )

        # Output projection (optional)
        if self._output_dim != hidden_channels:
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, self._output_dim),
                nn.LayerNorm(self._output_dim),
            )
        else:
            self.head = None

    def _make_blocks(
        self,
        hidden_channels,
        num_blocks,
        block_config,
        aggr,
        graph_aggr,
        norm,
        dropout,
        use_residual,
        nn,
    ):
        """Build sequence of GraphSAGE blocks."""
        blocks = nn.ModuleList()

        if block_config is None:
            block_config = {}

        if isinstance(block_config, dict):
            # Same configuration for all blocks
            for _ in range(num_blocks):
                blocks.append(
                    Block(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        aggr=aggr,
                        graph_aggr=graph_aggr,
                        norm=norm,
                        dropout=dropout,
                        use_residual=use_residual,
                        nn=nn,
                        **block_config,
                    )
                )
        elif isinstance(block_config, list):
            # Different configuration for each block
            for cfg in block_config:
                block_kwargs = {
                    "in_channels": hidden_channels,
                    "out_channels": hidden_channels,
                    "aggr": aggr,
                    "graph_aggr": graph_aggr,
                    "norm": norm,
                    "dropout": dropout,
                    "use_residual": use_residual,
                    "nn": nn,
                }
                block_kwargs.update(cfg)
                blocks.append(Block(**block_kwargs))
        else:
            raise ValueError(
                f"block_config must be dict or list, got {type(block_config)}"
            )

        return blocks

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            node_embed: Node embeddings [num_nodes, hidden_channels]
            graph_embed: Graph embeddings [num_graphs, hidden_channels]
        """
        assert info.get("x") is not None, "Input node features are required"
        assert info.get("edge_index") is not None, "Input edge indices are required"
        assert info.get("batch") is not None, "Input batch assignment is required"

        x, edge_index, batch = info["x"], info["edge_index"], info["batch"]

        # Initial projection
        node_embed = self.fc(x)
        graph_embed = None  # Initialize as None, will be created by first block

        # Residual blocks
        for block in self.blocks:
            node_embed, graph_embed = block(node_embed, edge_index, batch, graph_embed)

        # Optional output projection
        if self.head is not None:
            node_embed, graph_embed = self.head(node_embed), graph_embed

        info["node_embed"], info["graph_embed"] = node_embed, graph_embed

        return info

    @property
    def output_dim(self):
        """Output channels dimension."""
        return self._output_dim
