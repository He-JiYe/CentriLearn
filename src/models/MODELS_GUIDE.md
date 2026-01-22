# Models Module Guide

This guide explains how to use the models module following mmdet-style architecture.

## Architecture Overview

```
src/models/
├── __init__.py                 # Model registry and factory
├── MODELS_GUIDE.md           # This guide
├── backbones/                # Backbone networks
│   ├── __init__.py
│   ├── SimpleNet.py          # Simple single-branch network
│   ├── DeepNet.py           # Deep network with residual blocks
│   └── FPNet.py            # Feature Pyramid Network
├── nn/                      # Neural network building blocks
│   ├── __init__.py
│   └── GraphSAGE.py        # GraphSAGE implementation
├── network_dismantler/       # Network dismantling task models
│   ├── __init__.py
│   ├── ActorCritic.py       # Actor-Critic model
│   └── Qnet.py             # Q-network model
└── utils/                   # Model building utilities
    ├── __init__.py
    ├── registry.py          # Model registry
    └── builder.py           # Model builder
```

## Registry System

The module uses a registry system similar to mmdet for flexible model configuration:

```python
from src.models.utils.registry import BACKBONES, NETWORK_DISMANTLER

# Available registries:
# - BACKBONES: Graph neural network backbones (SimpleNet, DeepNet, FPNet)
# - NETWORK_DISMANTLER: Task-specific models for network dismantling (ActorCritic, Qnet)
```

## Quick Start

### 1. Build Model from Configuration

```python
from src.models.utils import build_backbone, build_network_dismantler

# Build backbone directly
backbone = build_backbone({
    'type': 'SimpleNet',
    'input_dim': 10,
    'hidden_dim': 64,
    'num_layers': 3
})

# Build task-specific model
model = build_network_dismantler({
    'type': 'ActorCritic',
    'backbone_cfg': {
        'type': 'SimpleNet',
        'input_dim': 10,
        'hidden_dim': 64
    }
})
```

### 2. Use Backbone Networks

```python
import torch

# Create dummy data
x = torch.randn(100, 10)          # [num_nodes, input_dim]
edge_index = torch.randint(0, 100, (2, 200))  # [2, num_edges]
batch = torch.zeros(100, dtype=torch.long)

# Build backbone
backbone = build_backbone({
    'type': 'SimpleNet',
    'input_dim': 10,
    'hidden_dim': 64
})

# Forward pass
output = backbone(x, edge_index, batch)
```

## Available Components

### Backbone Networks

| Backbone | Description | Key Features |
|----------|-------------|--------------|
| **SimpleNet** | Simple single-branch network | - Single GraphSAGE encoder<br>- Flexible layer configuration<br>- Graph embedding support |
| **DeepNet** | Deep network with residual blocks | - ResNet-style architecture<br>- Block-level residual connections<br>- Customizable per-block config |
| **FPNet** | Feature Pyramid Network | - Multi-scale features<br>- Feature fusion (TODO) |

### Task-Specific Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **ActorCritic** | Actor-Critic architecture | PPO, A2C algorithms |
| **Qnet** | Q-value network | DQN algorithm |

## Backbone Configuration

### SimpleNet

Simple backbone with a single GraphSAGE encoder.

```python
config = {
    'type': 'SimpleNet',
    'input_dim': 10,           # Input feature dimension
    'hidden_dim': 64,          # Hidden feature dimension
    'num_layers': 3,           # Number of GraphSAGE layers
    'output_dim': 64,          # Output dimension (default: hidden_dim)
    'aggr': 'mean',            # Node aggregation: 'mean', 'max', 'sum'
    'graph_aggr': 'add',       # Graph pooling: 'add', 'mean', 'max'
    'norm': 'layer',           # Normalization: 'layer', 'batch', None
    'dropout': 0.0             # Dropout probability
}
```

### DeepNet

Deep backbone with residual blocks (ResNet-style).

```python
config = {
    'type': 'DeepNet',
    'input_dim': 10,           # Input feature dimension
    'hidden_dim': 64,          # Hidden feature dimension
    'num_blocks': 3,           # Number of residual blocks
    'block_config': {...},      # Block configuration (optional)
    'aggr': 'mean',            # Node aggregation
    'graph_aggr': 'add',       # Graph pooling
    'norm': 'layer',           # Normalization
    'dropout': 0.0,           # Dropout
    'use_residual': True,       # Use residual connections
    'output_dim': 64           # Output dimension
}
```

**Block Configuration Options:**

```python
# Option 1: Same configuration for all blocks
block_config = {
    'norm': 'layer',
    'dropout': 0.1
}

# Option 2: Different configuration per block
block_config = [
    {'norm': 'layer', 'dropout': 0.1},  # Block 0
    {'norm': 'layer', 'dropout': 0.2},  # Block 1
    {'norm': 'batch', 'dropout': 0.2},   # Block 2
]
```

### FPNet

Feature Pyramid Network (under development).

```python
config = {
    'type': 'FPNet',
    # Configuration TBD
}
```

## Task Model Configuration

### ActorCritic

```python
config = {
    'type': 'ActorCritic',
    'backbone_cfg': {
        'type': 'SimpleNet',      # or 'DeepNet'
        'input_dim': 10,
        'hidden_dim': 64
    }
}
```

### Qnet

```python
config = {
    'type': 'Qnet',
    'backbone_cfg': {
        'type': 'SimpleNet',      # or 'DeepNet'
        'input_dim': 10,
        'hidden_dim': 64
    }
}
```

## Creating Custom Models

### Method 1: Register Custom Backbone

```python
import torch.nn as nn
from src.models.utils.registry import BACKBONES
from src.models.nn.GraphSAGE import GraphSAGE

@BACKBONES.register_module()
class MyCustomBackbone(nn.Module):
    """Custom backbone for specific task."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.conv = GraphSAGE(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            output_channels=hidden_dim
        )

    def forward(self, x, edge_index, batch, return_graph_embed=False):
        x = torch.nn.functional.relu(self.fc(x))
        x = self.conv(x, edge_index, batch, return_graph_embed)
        return x

# Use custom backbone
from src.models.utils import build_backbone
backbone = build_backbone({
    'type': 'MyCustomBackbone',
    'input_dim': 10,
    'hidden_dim': 128
})
```

### Method 2: Register Custom Task Model

```python
import torch.nn as nn
from src.models.utils.registry import NETWORK_DISMANTLER

@NETWORK_DISMANTLER.register_module()
class MyCustomTaskModel(nn.Module):
    """Custom task-specific model."""

    def __init__(self, backbone_cfg=None):
        super().__init__()

        if backbone_cfg is not None:
            from src.models.utils import build_backbone
            self.backbone = build_backbone(backbone_cfg)
        else:
            raise ValueError("backbone_cfg must be provided")

    def forward(self, x, **kwargs):
        return self.backbone(x, **kwargs)

# Use custom model
from src.models.utils import build_network_dismantler

model = build_network_dismantler({
    'type': 'MyCustomTaskModel',
    'backbone_cfg': {
        'type': 'SimpleNet',
        'input_dim': 10,
        'hidden_dim': 64
    }
})
```

### Method 3: Extend Existing Backbones

```python
from src.models.backbones import SimpleNet

class MyEnhancedNet(SimpleNet):
    """Extended SimpleNet with additional features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom components
        self.extra_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, edge_index, batch, return_graph_embed=False):
        x = super().forward(x, edge_index, batch, return_graph_embed)
        # Add custom logic
        x = self.extra_layer(x)
        return x
```

## Building Models from Config Files

### Example: Network Dismantling with ActorCritic

```yaml
# configs/network_dismantling/actor_critic_simple.yaml
model:
  type: ActorCritic
  backbone_cfg:
    type: SimpleNet
    input_dim: 10
    hidden_dim: 64
    num_layers: 3
    aggr: mean
    graph_aggr: add
    norm: layer
    dropout: 0.0
```

### Example: Deep Network with Custom Blocks

```yaml
# configs/network_dismantling/actor_critic_deep.yaml
model:
  type: ActorCritic
  backbone_cfg:
    type: DeepNet
    input_dim: 10
    hidden_dim: 128
    num_blocks: 4
    block_config:
      - norm: layer
        dropout: 0.1
      - norm: layer
        dropout: 0.1
      - norm: batch
        dropout: 0.2
      - norm: batch
        dropout: 0.2
    use_residual: true
```

### Loading from YAML

```python
import yaml
from src.models.utils import build_network_dismantler

# Load config
with open('configs/network_dismantling/actor_critic_simple.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
model = build_network_dismantler(config['model'])

# Use model
x = torch.randn(100, 10)
edge_index = torch.randint(0, 100, (2, 200))
batch = torch.zeros(100, dtype=torch.long)

output = model(x, edge_index=edge_index, batch=batch)
```

## API Reference

### Build Functions

```python
from src.models.utils import build_backbone, build_network_dismantler

# Build backbone
backbone = build_backbone(config_dict)

# Build task model
model = build_network_dismantler(config_dict)
```

### Registry Access

```python
from src.models.utils.registry import BACKBONES, NETWORK_DISMANTLER

# Get registered class
backbone_class = BACKBONES.get('SimpleNet')
model_class = NETWORK_DISMANTLER.get('ActorCritic')

# Show all registered items
print(list(BACKBONES.module_dict.keys()))        # ['SimpleNet', 'DeepNet', 'FPNet']
print(list(NETWORK_DISMANTLER.module_dict.keys()))  # ['ActorCritic', 'Qnet']

# Check if registered
'SimpleNet' in BACKBONES  # True
'ActorCritic' in NETWORK_DISMANTLER  # True
```

## Best Practices

1. **Use Registry**: Always register custom models using `@BACKBONES.register_module()` or `@NETWORK_DISMANTLER.register_module()`
2. **Flexible Config**: Use YAML configs for easy experimentation
3. **Modular Design**: Separate backbones, task models, and utility functions
4. **Documentation**: Document custom models with comprehensive docstrings
5. **Testing**: Test models with dummy data before real training
6. **Backbone Reuse**: Task models should compose backbones rather than reimplementing logic
7. **Type Hints**: Use Python type hints for better IDE support

## Common Patterns

### Backbone with Graph Embedding

```python
class MyBackbone(nn.Module):
    def forward(self, x, edge_index, batch, return_graph_embed=False):
        # ... encode nodes ...
        if return_graph_embed:
            from torch_geometric.nn import global_mean_pool
            graph_embed = global_mean_pool(x, batch)
            return x, graph_embed
        return x
```

### Task Model with Multiple Outputs

```python
@NETWORK_DISMANTLER.register_module()
class MyTaskModel(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)

    def forward(self, x, edge_index, batch, **kwargs):
        # Get node embeddings
        node_embed, graph_embed = self.backbone(
            x, edge_index, batch, return_graph_embed=True
        )

        # Process outputs
        output1 = self.head1(node_embed)
        output2 = self.head2(graph_embed)

        return output1, output2
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to import the model before building
   ```python
   from src.models.backbones import SimpleNet  # Import to register
   from src.models.utils import build_backbone

   backbone = build_backbone({'type': 'SimpleNet', ...})
   ```

2. **Registry Key Error**: Check if model is registered
   ```python
   print(list(BACKBONES.module_dict.keys()))  # Show all registered backbones
   print(list(NETWORK_DISMANTLER.module_dict.keys()))  # Show task models
   ```

3. **Dimension Mismatch**: Ensure input_dim matches your data
   ```python
   x.shape[1] == config['input_dim']  # Should be True
   ```

4. **Missing Arguments**: Verify all required config keys are provided
   ```python
   # SimpleNet requires: type, input_dim
   # DeepNet requires: type, input_dim, hidden_dim, num_blocks
   # ActorCritic requires: type, backbone_cfg
   ```

## Examples

See `examples/` directory for complete examples:
- `build_model_example.py`: Building models from configs
- `custom_model_example.py`: Creating and registering custom models
