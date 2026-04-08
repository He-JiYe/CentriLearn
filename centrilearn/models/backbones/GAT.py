"""
The implementation of GAT with graph embedding based on PyG's GATConv.
"""

from typing import Any, Dict, Tuple, Union

from torch_geometric.nn import GATConv, GATv2Conv

from centrilearn.models.backbones.BasicGNN import BasicGNN
from centrilearn.utils.registry import BACKBONES


@BACKBONES.register_module()
class GAT(BasicGNN):
    """GAT with graph embedding computation.

    Uses PyG's GATConv for node-level message passing and computes
    graph-level embeddings via global pooling.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        num_layers: Number of GAT layers
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
        out_channels: Union[int, None] = None,
        aggr: str = "mean",
        graph_aggr: str = "add",
        norm: Union[str, None] = None,
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

    def init_conv(
        self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs
    ):
        v2 = kwargs.pop("v2", False)
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, "_is_conv_to_out", False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GATConv' (got '{out_channels}') is divisible "
                f"by the number of heads (got '{heads}')"
            )

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            dropout=self.dropout.p,
            **kwargs,
        )

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
    def output_dim(self) -> int:
        """Output channels dimension."""
        return self.out_channels
