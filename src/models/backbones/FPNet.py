"""
Feature Pyramid Network backbone for multi-scale graph features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn.GraphSAGE import GraphSAGE
from ..utils.registry import BACKBONES


@BACKBONES.register_module()
class FPNet(nn.Module):
    """Feature Pyramid Network for graph data.

    Builds multi-scale features using GraphSAGE encoders at different depths,
    then fuses them into a unified representation.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden dimensions for each pyramid level.
            Example: [64, 128, 256] for 3-level pyramid.
        num_layers_list: List of number of GraphSAGE layers for each pyramid level.
            If not provided, uses [1, 2, 3] progressively deeper.
        aggr: GraphSAGE aggregation method.
        graph_aggr: Graph pooling method.
        norm: Normalization type.
        dropout: Dropout probability.
        fusion_mode: Feature fusion method ('add', 'concat', 'attention').
        output_dim: Output feature dimension (default: max(hidden_dims)).
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [64, 128, 256],
                 num_layers_list: list = None,
                 aggr: str = 'mean',
                 graph_aggr: str = 'add',
                 norm: str = 'layer',
                 dropout: float = 0.0,
                 fusion_mode: str = 'add',
                 output_dim: int = None,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.fusion_mode = fusion_mode
        self.num_levels = len(hidden_dims)
        self.output_dim = output_dim if output_dim else max(hidden_dims)

        # Determine number of layers for each pyramid level
        if num_layers_list is None:
            num_layers_list = list(range(1, self.num_levels + 1))
        elif len(num_layers_list) != self.num_levels:
            raise ValueError(
                f"num_layers_list length ({len(num_layers_list)}) "
                f"must match hidden_dims length ({self.num_levels})"
            )

        # Build pyramid levels
        self.pyramid = nn.ModuleList()
        prev_dim = input_dim

        for i, (hidden_dim, num_layers) in enumerate(zip(hidden_dims, num_layers_list)):
            # Projection for this level
            self.pyramid.append(nn.Linear(prev_dim, hidden_dim))

            # GraphSAGE encoder with different depth for each level
            self.pyramid.append(
                GraphSAGE(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    num_layers=num_layers,  
                    output_channels=hidden_dim,
                    aggr=aggr,
                    graph_aggr=graph_aggr,
                    norm=norm,
                    dropout=dropout,
                    **kwargs
                )
            )

            prev_dim = hidden_dim

        # Feature fusion layer
        if fusion_mode == 'concat':
            fusion_input_dim = sum(hidden_dims)
        else:  # 'add' or 'attention'
            fusion_input_dim = max(hidden_dims)

            # Projection layers for alignment in 'add' mode
            if fusion_mode == 'add':
                self.level_projections = nn.ModuleList()
                for hidden_dim in hidden_dims:
                    if hidden_dim < fusion_input_dim:
                        self.level_projections.append(
                            nn.Linear(hidden_dim, fusion_input_dim)
                        )
                    else:
                        self.level_projections.append(nn.Identity())

        if fusion_mode == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_input_dim // 4, len(hidden_dims)),
                nn.Softmax(dim=-1)
            )

        # Output projection
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(inplace=True)
        )

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
        # Extract multi-scale features at different depths
        x_features, graph_features = [], []

        # Iterate through pyramid: (proj, graphsage)
        for i in range(0, len(self.pyramid), 2):
            # Projection
            x_proj = self.pyramid[i](x)

            # Graph encoding with different depth
            x_feat, graph_feat = self.pyramid[i + 1](x_proj, edge_index, batch)
            x_features.append(x_feat)
            graph_features.append(graph_feat)

        # Feature fusion
        if self.fusion_mode == 'concat':
            x_fused, graph_fused = torch.cat(x_features, dim=-1), torch.cat(graph_features, dim=-1)
        elif self.fusion_mode == 'add':
            x_projected = [self.level_projections[i](feat) for i, feat in enumerate(x_features)]
            graph_projected = [self.level_projections[i](feat) for i, feat in enumerate(graph_features)]
            x_fused, graph_fused = torch.stack(x_projected, dim=0).sum(dim=0), torch.stack(graph_projected, dim=0).sum(dim=0)
        elif self.fusion_mode == 'attention':
            x_stacked, graph_stacked = torch.stack(x_features, dim=0), torch.stack(graph_features, dim=0) # [num_levels, num_nodes, dim]
            x_attention_weights, graph_attention_weights = self.attention(x_stacked.mean(dim=-1)), self.attention(graph_stacked.mean(dim=-1))  # [num_nodes, num_levels]
            x_attention_weights, graph_attention_weights = x_attention_weights.permute(1, 0, 2).unsqueeze(-1), graph_attention_weights.permute(1, 0, 2).unsqueeze(-1)   # [num_levels, num_nodes, 1]
            x_fused, graph_fused = (x_stacked * x_attention_weights).sum(dim=0), (graph_stacked * graph_attention_weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Output projection
        node_output, graph_output = self.fusion(x_fused), self.fusion(graph_fused)

        return node_output, graph_output

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

        # Extract multi-scale features at different depths
        x_features, graph_features = [], []

        # Iterate through pyramid: (proj, graphsage)
        for i in range(0, len(self.pyramid), 2):
            # Projection
            x_proj = self.pyramid[i](x)

            # Graph encoding with different depth
            x_feat, graph_feat = self.pyramid[i + 1](x_proj, edge_index, batch)
            x_features.append(x_feat)
            graph_features.append(graph_feat)

        # Feature fusion
        if self.fusion_mode == 'concat':
            x_fused, graph_fused = torch.cat(x_features, dim=-1), torch.cat(graph_features, dim=-1)
        elif self.fusion_mode == 'add':
            x_projected = [self.level_projections[i](feat) for i, feat in enumerate(x_features)]
            graph_projected = [self.level_projections[i](feat) for i, feat in enumerate(graph_features)]
            x_fused, graph_fused = torch.stack(x_projected, dim=0).sum(dim=0), torch.stack(graph_projected, dim=0).sum(dim=0)
        elif self.fusion_mode == 'attention':
            x_stacked, graph_stacked = torch.stack(x_features, dim=0), torch.stack(graph_features, dim=0) # [num_levels, num_nodes, dim]
            x_attention_weights, graph_attention_weights = self.attention(x_stacked.mean(dim=-1)), self.attention(graph_stacked.mean(dim=-1))  # [num_nodes, num_levels]
            x_attention_weights, graph_attention_weights = x_attention_weights.permute(1, 0, 2).unsqueeze(-1), graph_attention_weights.permute(1, 0, 2).unsqueeze(-1)   # [num_levels, num_nodes, 1]
            x_fused, graph_fused = (x_stacked * x_attention_weights).sum(dim=0), (graph_stacked * graph_attention_weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Output projection
        node_output, graph_output = self.fusion(x_fused), self.fusion(graph_fused)

        # Update info
        info['node_embed'] = node_output
        info['graph_embed'] = graph_output

        return info

    @property
    def out_channels(self):
        """Output channels dimension."""
        return self.output_dim
