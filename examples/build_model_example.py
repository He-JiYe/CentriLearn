"""
Example: Building models from configuration files.
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import build_model


def build_from_yaml(config_path: str):
    """Build model from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = build_model(config['model'])
    return model


def main():
    """Build models from different configurations."""

    # Example 1: Build from YAML file
    print("=" * 60)
    print("Example 1: Build from YAML configuration")
    print("=" * 60)

    config_path = "configs/network_dismantling/finder_actor_critic.yaml"
    model = build_from_yaml(config_path)

    print(f"Model type: {type(model).__name__}")
    print(f"Backbone type: {type(model.backbone).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example 2: Build from Python dict
    print("\n" + "=" * 60)
    print("Example 2: Build from Python dictionary")
    print("=" * 60)

    config = {
        'type': 'FINDER',
        'backbone': {
            'type': 'QNet',
            'input_dim': 20,
            'hidden_dim': 128,
            'num_layers': 3,
            'aggr': 'max',
            'graph_aggr': 'mean',
            'norm': 'layer',
            'dropout': 0.1,
            'use_residual': True
        }
    }

    model = build_model(config)
    print(f"Model type: {type(model).__name__}")
    print(f"Backbone type: {type(model.backbone).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example 3: Build backbone directly
    print("\n" + "=" * 60)
    print("Example 3: Build backbone directly")
    print("=" * 60)

    from src.models.utils import build_backbone

    backbone_config = {
        'type': 'ActorCritic',
        'input_dim': 10,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_critics': 2
    }

    backbone = build_backbone(backbone_config)
    print(f"Backbone type: {type(backbone).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in backbone.parameters()):,}")

    # Example 4: Test forward pass
    print("\n" + "=" * 60)
    print("Example 4: Test forward pass")
    print("=" * 60)

    import torch
    from torch_geometric.data import Batch

    # Create dummy data
    num_nodes = 100
    input_dim = 10
    batch_size = 4

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.randint(0, batch_size, (num_nodes,))
    branch = torch.randint(0, 3, (num_nodes,))

    # Build ActorCritic backbone
    backbone = build_backbone({
        'type': 'ActorCritic',
        'input_dim': input_dim,
        'hidden_dim': 64
    })

    logit, value, features = backbone(x, edge_index, batch, branch)

    print(f"Input features shape: {x.shape}")
    print(f"Output logits shape: {logit.shape}")
    print(f"Output values shape: {value.shape}")
    print(f"Output features shape: {features.shape}")


if __name__ == '__main__':
    main()
