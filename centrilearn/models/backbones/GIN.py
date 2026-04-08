"""
The implementation of GIN with graph embedding based on PyG's GINConv.
"""

from typing import Any, Dict

from torch_geometric.nn import MLP, GINConv

from centrilearn.models.backbones.BasicGNN import BasicGNN
from centrilearn.utils.registry import BACKBONES


@BACKBONES.register_module()
class GIN(BasicGNN):
    """GIN with graph embedding computation.

    Uses PyG's GINConv for node-level message passing and computes
    graph-level embeddings via global pooling.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        num_layers: Number of GIN layers
        out_channels: Output feature dimension (default: hidden_channels)
        graph_aggr: Graph pooling method ('add', 'mean', 'max')
        dropout: Dropout probability
    """

    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int = None,
        aggr: str = "mean",
        graph_aggr: str = "add",
        norm: str = None,
        dropout: float = 0.0,
        fpa: bool = False,
        **kwargs,
    ):
        self.aggr = aggr

        # Initialize BasicGNN with graph_aggr option and fpa support
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            norm=norm,
            graph_aggr=graph_aggr,
            fpa=fpa,
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

    def forward(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            node_embed: Node embeddings [num_nodes, hidden_channels]
            graph_embed: Graph embeddings [num_graphs, hidden_channels]
        """
        assert info.get("x") is not None, "x is required"
        assert info.get("edge_index") is not None, "Edge indices are required"
        assert info.get("batch") is not None, "Batch assignment is required"

        x, edge_index, batch = info["x"], info["edge_index"], info["batch"]
        edge_weight, edge_attr, batch_size = (
            info.get("edge_weight"),
            info.get("edge_attr"),
            info.get("batch_size"),
        )

        # Call parent's forward which returns (node_embed, graph_embed)
        node_embed, graph_embed = super().forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
        )

        info["node_embed"], info["graph_embed"] = node_embed, graph_embed
        return info

    @property
    def output_dim(self):
        """Output channels dimension."""
        return self.out_channels
