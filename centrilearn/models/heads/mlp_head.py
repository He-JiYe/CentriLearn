"""
MLP head for various prediction tasks.
"""
from torch import nn, Tensor
from centrilearn.utils.registry import HEADS


@HEADS.register_module()
class MLPHead(nn.Module):
    """Generic MLP head for prediction tasks.

    Args:
        in_channels: Input feature dimension.
        hidden_layers: List of hidden layer dimensions. Example: [128, 64, 1].
        activation: Activation function ('relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'none').
        dropout: Dropout probability.
        norm: Normalization type ('batch', 'layer', 'none').
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: list = None,
                 activation: str = 'leaky_relu',
                 dropout: float = 0.0,
                 norm: str = 'none'):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_channels, 1]

        self.activation_name = activation
        self.dropout = dropout
        self.norm_type = norm

        # Build MLP layers
        layers = []
        current_dim = in_channels

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Hidden layers: activation + norm + dropout
            if i < len(hidden_layers) - 1:
                if norm == 'batch':
                    layers.append(nn.BatchNorm1d(hidden_dim))
                elif norm == 'layer':
                    layers.append(nn.LayerNorm(hidden_dim))

                if activation != 'none':
                    act_map = {
                        'relu': nn.ReLU,
                        'leaky_relu': nn.LeakyReLU,
                        'tanh': nn.Tanh,
                        'sigmoid': nn.Sigmoid,
                        'gelu': nn.GELU
                    }
                    act = act_map.get(activation, nn.LeakyReLU)
                    layers.append(act())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass (legacy compatibility).

        Args:
            x: Input features [..., in_channels]

        Returns:
            Output features [..., output_dim]
        """
        return self.mlp(x)
