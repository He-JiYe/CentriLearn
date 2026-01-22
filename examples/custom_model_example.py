"""
Example: Creating custom models and registering them.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from src.models.nn.GraphSAGE import GraphSAGE
from src.models.utils import MODELS, BACKBONES, build_model


# Example 1: Register a custom backbone
@BACKBONES.register_module()
class CustomGNN(nn.Module):
    """Custom GNN backbone for demonstration."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.conv = GraphSAGE(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            output_channels=hidden_dim
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.fc(x)
        x = F.relu(x)
        x = self.conv(x, edge_index, batch)
        x = self.head(x)
        return x


# Example 2: Register a custom task-specific model
@MODELS.register_module()
class CustomTaskModel(nn.Module):
    """Custom task-specific model."""

    def __init__(self, backbone=None):
        super().__init__()
        if backbone is not None:
            from src.models.utils import build_backbone
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None

    def forward(self, x, edge_index, batch, **kwargs):
        if self.backbone is not None:
            return self.backbone(x, edge_index, batch, **kwargs)
        return x


def main():
    """Demonstrate custom model registration and usage."""

    print("=" * 60)
    print("Example 1: Build custom registered backbone")
    print("=" * 60)

    # Build custom backbone using config
    config = {
        'type': 'CustomGNN',
        'input_dim': 10,
        'hidden_dim': 128,
        'num_layers': 3
    }

    from src.models.utils import build_backbone
    model = build_backbone(config)

    print(f"Model type: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)

    output = model(x, edge_index, batch)
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("Example 2: Build custom registered model")
    print("=" * 60)

    # Build custom task model
    model_config = {
        'type': 'CustomTaskModel',
        'backbone': {
            'type': 'ActorCritic',
            'input_dim': 10,
            'hidden_dim': 64
        }
    }

    model = build_model(model_config)
    print(f"Model type: {type(model).__name__}")
    print(f"Backbone type: {type(model.backbone).__name__}")

    # Test forward pass
    branch = torch.randint(0, 3, (100,))
    logit, value, features = model(x, edge_index, batch, branch)
    print(f"Logit shape: {logit.shape}")
    print(f"Value shape: {value.shape}")

    print("\n" + "=" * 60)
    print("Available models and backbones")
    print("=" * 60)
    print(f"Registered MODELS: {list(MODELS.module_dict.keys())}")
    print(f"Registered BACKBONES: {list(BACKBONES.module_dict.keys())}")


if __name__ == '__main__':
    main()
