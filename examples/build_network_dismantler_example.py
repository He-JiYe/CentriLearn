"""
Example: Building network_dismantler models from configuration files.

This example demonstrates how to build network dismantling models
using backbone + head architecture with configuration files.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from src.models import build_backbone, build_head, build_network_dismantler


def load_and_build_model(config_path: str):
    """Load config and build model."""
    print(f"\n{'=' * 60}")
    print(f"Loading config: {config_path}")
    print('=' * 60)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_network_dismantler(config['model'])

    print(f"Model type: {type(model).__name__}")
    print(f"Backbone type: {type(model.backbone).__name__}")

    # Show head information if available
    if hasattr(model, 'actor'):
        print(f"Actor head type: {type(model.actor).__name__}")
    if hasattr(model, 'critics') and model.critics:
        print(f"Critic head type: {type(model.critics[0]).__name__}")
        print(f"Number of critics: {len(model.critics)}")
    if hasattr(model, 'q_head'):
        print(f"Q-head type: {type(model.q_head).__name__}")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, config


def test_actor_critic():
    """Test ActorCritic model."""
    print("\n" + "=" * 60)
    print("Testing ActorCritic Model")
    print("=" * 60)

    # Build model
    config_path = "configs/network_dismantling/actor_critic_simplenet.yaml"
    model, config = load_and_build_model(config_path)

    # Create dummy data
    num_nodes = 100
    input_dim = config['model']['backbone_cfg']['input_dim']
    batch_size = 4

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.randint(0, batch_size, (num_nodes,))
    branch = torch.randint(0, 3, (num_nodes,))

    # Forward pass with branch aggregation
    print("\n--- Forward pass with branch aggregation ---")
    output = model(x, edge_index, batch, branch=branch)

    print(f"Input features shape: {x.shape}")
    print(f"Output logit shape: {output['logit'].shape}")
    print(f"Output value shape: {output['value'].shape}")
    print(f"Node embed shape: {output['node_embed'].shape}")
    print(f"Graph embed shape: {output['graph_embed'].shape}")

    # Forward pass without branch aggregation
    print("\n--- Forward pass without branch aggregation ---")
    output = model(x, edge_index, batch, branch=None)

    print(f"Output logit shape: {output['logit'].shape}")
    print(f"Critic values count: {len(output['value'])}")
    for i, critic_value in enumerate(output['value']):
        print(f"  Critic {i} shape: {critic_value.shape}")

    # Action selection
    print("\n--- Action selection ---")
    action, logit = model.get_action(x, edge_index, batch)
    print(f"Selected actions shape: {action.shape}")
    print(f"Sample actions: {action[:5].tolist()}")

    # Value estimation
    print("\n--- Value estimation ---")
    value = model.get_value(x, edge_index, batch, branch)
    print(f"Estimated values shape: {value.shape}")


def test_qnet():
    """Test Qnet model."""
    print("\n" + "=" * 60)
    print("Testing Qnet Model")
    print("=" * 60)

    # Build model
    config_path = "configs/network_dismantling/qnet_simplenet.yaml"
    model, config = load_and_build_model(config_path)

    # Create dummy data
    num_nodes = 100
    input_dim = config['model']['backbone_cfg']['input_dim']
    batch_size = 4

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.randint(0, batch_size, (num_nodes,))

    # Forward pass
    print("\n--- Forward pass ---")
    output = model(x, edge_index, batch)

    print(f"Input features shape: {x.shape}")
    print(f"Q-values shape: {output['q_values'].shape}")
    print(f"Node embed shape: {output['node_embed'].shape}")
    print(f"Graph embed shape: {output['graph_embed'].shape}")

    # Q-value prediction
    print("\n--- Q-value prediction ---")
    q_values = model.get_q_values(x, edge_index, batch)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[:5].flatten().tolist()}")

    # Action selection (greedy)
    print("\n--- Greedy action selection ---")
    action, q_value = model.select_action(x, edge_index, batch, epsilon=0.0)
    print(f"Selected actions: {action[:5].tolist()}")
    print(f"Sample Q-values: {q_value[:5].flatten().tolist()}")

    # Action selection (epsilon-greedy)
    print("\n--- Epsilon-greedy action selection (epsilon=0.5) ---")
    action, q_value = model.select_action(x, edge_index, batch, epsilon=0.5)
    print(f"Selected actions: {action[:5].tolist()}")
    print(f"Sample Q-values: {q_value[:5].flatten().tolist()}")

    # Loss computation
    print("\n--- Loss computation ---")
    target_q_values = torch.randn_like(q_values)
    loss = model.compute_loss(q_values, target_q_values)
    print(f"Q-learning loss: {loss.item():.4f}")


def test_multiple_configs():
    """Test multiple configurations."""
    print("\n" + "=" * 60)
    print("Testing Multiple Configurations")
    print("=" * 60)

    configs = [
        "configs/network_dismantling/actor_critic_simplenet.yaml",
        "configs/network_dismantling/actor_critic_deepnet.yaml",
        "configs/network_dismantling/qnet_simplenet.yaml",
        "configs/network_dismantling/qnet_fpnet.yaml"
    ]

    for config_path in configs:
        try:
            model, _ = load_and_build_model(config_path)
        except Exception as e:
            print(f"Error loading {config_path}: {e}")


def test_building_backbone_and_head():
    """Test building backbone and head separately."""
    print("\n" + "=" * 60)
    print("Testing Backbone + Head Separate Building")
    print("=" * 60)

    # Build backbone
    backbone_cfg = {
        'type': 'SimpleNet',
        'input_dim': 10,
        'hidden_dim': 64,
        'num_layers': 3,
        'aggr': 'mean',
        'graph_aggr': 'add',
        'norm': 'layer',
        'dropout': 0.0
    }

    backbone = build_backbone(backbone_cfg)
    print(f"Backbone type: {type(backbone).__name__}")
    print(f"Backbone out_channels: {backbone.out_channels}")

    # Build actor head
    actor_head_cfg = {
        'type': 'LogitHead',
        'in_channels': 64,
        'hidden_layers': [64, 64, 1],
        'activation': 'leaky_relu'
    }

    actor = build_head(actor_head_cfg)
    print(f"Actor head type: {type(actor).__name__}")

    # Build critic head
    critic_head_cfg = {
        'type': 'VHead',
        'in_channels': 64,
        'hidden_layers': [64, 64, 1],
        'activation': 'leaky_relu'
    }

    critic = build_head(critic_head_cfg)
    print(f"Critic head type: {type(critic).__name__}")

    # Test forward pass
    num_nodes = 100
    input_dim = 10
    batch_size = 4

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.randint(0, batch_size, (num_nodes,))

    node_embed, graph_embed = backbone(x, edge_index, batch)
    logit = actor(node_embed)
    value = critic(node_embed)

    print(f"\n--- Separate Backbone + Head Forward Pass ---")
    print(f"Input features shape: {x.shape}")
    print(f"Node embed shape: {node_embed.shape}")
    print(f"Graph embed shape: {graph_embed.shape}")
    print(f"Logit shape: {logit.shape}")
    print(f"Value shape: {value.shape}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Network Dismantler Model Building Examples")
    print("=" * 60)

    # Test building backbone and head separately
    test_building_backbone_and_head()

    # Test ActorCritic
    test_actor_critic()

    # Test Qnet
    test_qnet()

    # Test multiple configs
    test_multiple_configs()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
