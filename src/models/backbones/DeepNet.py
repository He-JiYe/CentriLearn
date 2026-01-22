"""
Deep network backbone with residual connections for graph reinforcement learning.
"""
import torch.nn as nn
from ..nn.GraphSAGE import GraphSAGE
from ..utils.registry import BACKBONES


class GraphSAGEBlock(nn.Module):
    """A single GraphSAGE block with optional residual connection.

    Similar to ResNet block, this block wraps a single-layer GraphSAGE
    and applies normalization, activation, and optional residual connection.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        aggr: GraphSAGE aggregation method.
        graph_aggr: Graph pooling method.
        norm: Normalization type.
        dropout: Dropout probability.
        use_residual: Whether to use residual connection.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 aggr: str = 'mean',
                 graph_aggr: str = 'add',
                 norm: str = 'layer',
                 dropout: float = 0.0,
                 use_residual: bool = True):
        super().__init__()

        self.use_residual = use_residual
        self.channel_match = (in_channels == out_channels)

        # Single-layer GraphSAGE
        self.conv = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=out_channels,
            num_layers=1,
            output_channels=out_channels,
            aggr=aggr,
            graph_aggr=graph_aggr,
            norm=norm,
            dropout=dropout
        )

        # Normalization
        if norm == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'batch':
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

    def forward(self, x, edge_index, batch, graph_embed=None):
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            graph_embed: Graph embedding [num_graphs, in_channels] (Default: None)

        Returns:
            Updated node features and graph embedding [num_nodes, out_channels]
        """
        node_identity, graph_identity = x, graph_embed

        # Graph convolution
        node_out, graph_out = self.conv(x, edge_index, batch, graph_embed, return_graph_embed=True)

        # Residual connection
        if self.use_residual:
            if not self.channel_match:
                node_identity, graph_identity = self.proj(node_identity), self.proj(graph_identity)
            node_out, graph_out = node_out + node_identity, graph_out + graph_identity

        # Normalization and activation
        node_out, graph_out = self.norm(node_out), self.norm(graph_out)
        node_out, graph_out = self.act(node_out), self.norm(graph_out)

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

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_blocks: int = 3,
                 block_config: dict = None,
                 aggr: str = 'mean',
                 graph_aggr: str = 'add',
                 norm: str = 'layer',
                 dropout: float = 0.0,
                 use_residual: bool = True,
                 output_dim: int = None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else hidden_dim
        self.use_residual = use_residual

        # Node to embedding
        self.fc = nn.Linear(input_dim, hidden_dim),

        # Build blocks
        self.blocks = self._make_blocks(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_config=block_config,
            aggr=aggr,
            graph_aggr=graph_aggr,
            norm=norm,
            dropout=dropout,
            use_residual=use_residual
        )

        # Output projection (optional)
        if self.output_dim != hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.head = None

    def _make_blocks(self, hidden_dim, num_blocks, block_config,
                     aggr, graph_aggr, norm, dropout, use_residual):
        """Build sequence of GraphSAGE blocks."""
        blocks = nn.ModuleList()

        if block_config is None:
            block_config = {}

        if isinstance(block_config, dict):
            # Same configuration for all blocks
            for _ in range(num_blocks):
                blocks.append(
                    GraphSAGEBlock(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        aggr=aggr,
                        graph_aggr=graph_aggr,
                        norm=norm,
                        dropout=dropout,
                        use_residual=use_residual,
                        **block_config
                    )
                )
        elif isinstance(block_config, list):
            # Different configuration for each block
            for cfg in block_config:
                block_kwargs = {
                    'in_channels': hidden_dim,
                    'out_channels': hidden_dim,
                    'aggr': aggr,
                    'graph_aggr': graph_aggr,
                    'norm': norm,
                    'dropout': dropout,
                    'use_residual': use_residual
                }
                block_kwargs.update(cfg)
                blocks.append(GraphSAGEBlock(**block_kwargs))
        else:
            raise ValueError(f"block_config must be dict or list, got {type(block_config)}")

        return blocks

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass (legacy compatibility).

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            **kwargs: optional arguments for GraphSAGE.
            
        Returns:
            Node features [num_nodes, output_dim]
            Graph embedding [num_graphs, output_dim]
        """
        # Initial projection
        node_embed, graph_embed = self.fc(x), None

        # Residual blocks
        for block in self.blocks:
            node_embed, graph_embed = block(node_embed, edge_index, batch, graph_embed)

        # Optional output projection
        if self.head is not None:
            node_embed, graph_embed = self.head(node_embed), graph_embed

        return node_embed, graph_embed

    def forward_info(self, info: dict):
        """Forward pass using info dictionary.

        Args:
            info: Dictionary containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]

        Returns:
            Dictionary containing:
                - node_embed: Node features [num_nodes, output_dim]
                - graph_embed: Graph embedding [num_graphs, output_dim]
        """
        x = info['x']
        edge_index = info['edge_index']
        batch = info['batch']

        # Initial projection
        node_embed, graph_embed = self.fc(x), None

        # Residual blocks
        for block in self.blocks:
            node_embed, graph_embed = block(node_embed, edge_index, batch, graph_embed)

        # Optional output projection
        if self.head is not None:
            node_embed, graph_embed = self.head(node_embed), graph_embed

        # Update info
        info['node_embed'] = node_embed
        info['graph_embed'] = graph_embed

        return info

    @property
    def out_channels(self):
        """Output channels dimension."""
        return self.output_dim
