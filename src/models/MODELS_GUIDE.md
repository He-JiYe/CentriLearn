# Models Module Guide

This guide explains how to use the models module following mmdet-style architecture.

## Architecture Overview

```
src/models/
├── __init__.py                 # Model registry and factory
├── MODELS_GUIDE.md             # This guide
├── nn/                         # Neural network building blocks
│   ├── __init__.py
│   └── GraphSAGE.py            # GraphSAGE implementation
├── backbones/                  # Backbone networks
│   ├── __init__.py
│   ├── SimpleNet.py            # Simple single-branch network
│   ├── DeepNet.py              # Deep network with residual blocks
│   └── FPNet.py                # Feature Pyramid Network
├── heads/                      # Prediction heads
│   ├── __init__.py
│   ├── mlp_head.py             # Generic MLP head
│   ├── q_head.py               # Q-value head
│   ├── v_head.py               # Value head (critic)
│   ├── logit_head.py           # Logit head (policy)
│   └── branch_value.py         # Branch value head for hierarchical tasks
├── network_dismantler/         # Network dismantling task models
│   ├── __init__.py
│   ├── ActorCritic.py          # Actor-Critic model
│   └── Qnet.py                 # Q-network model
└── utils/                      # Model building utilities
    ├── __init__.py
    ├── registry.py             # Model registry
    └── builder.py              # Model builder
```

## Registry System

The module uses a registry system similar to mmdet for flexible model configuration:

```python
from src.models.utils.registry import BACKBONES, HEADS, NETWORK_DISMANTLER

# Available registries:
# - BACKBONES: Graph neural network backbones (SimpleNet, DeepNet, FPNet)
# - HEADS: Prediction heads (MLPHead, QHead, VHead, LogitHead, BranchValueHead)
# - NETWORK_DISMANTLER: Task-specific models for network dismantling (ActorCritic, Qnet)
```

## Design Philosophy

The module follows a **backbone + head** architecture:
- **Backbones**: Extract features from graph data (node embeddings, graph embeddings)
- **Heads**: Make task-specific predictions (Q-values, values, logits)
- **Task Models**: Compose backbones and heads for specific algorithms

## Info Dictionary

All modules support both legacy positional arguments and flexible `info` dictionary passing:

### Info Dictionary Structure

```python
info = {
    'x': node_features,                 # Node features [num_nodes, input_dim]
    'edge_index': edge_indices,         # Edge indices [2, num_edges]
    'batch': batch_assignment,          # Batch assignment [num_nodes]
    'branch': branch_assignment,        # Optional: Branch assignment [num_nodes]
    'node_embed': node_embeddings,      # Optional: Added by backbone
    'graph_embed': graph_embeddings,    # Optional: Added by backbone
    # ... additional keys can be added as needed
}
```

### Using Info Dictionary

```python
# Backbone forward with info
info = {
    'x': x,
    'edge_index': edge_index,
    'batch': batch
}
info = backbone.forward_info(info)
# Now info contains 'node_embed' and 'graph_embed'

# Head forward with info
info = head.forward_info(info)
# Now info contains prediction results

# Task model forward with info
info = model.forward_info(info)
# Returns updated info with all predictions
```

### Legacy Compatibility

For backward compatibility, all modules also support positional arguments:

```python
# Legacy forward (still works)
node_embed, graph_embed = backbone(x, edge_index, batch)
logit = actor(node_embed)
value = critic(graph_embed)

# Info dictionary forward (recommended)
info = {'x': x, 'edge_index': edge_index, 'batch': batch}
info = backbone.forward_info(info)
info = actor.forward_info(info)
```

## Quick Start

### 1. Build Model from Configuration

```python
from src.models import build_backbone, build_head, build_network_dismantler

# Build backbone directly
backbone = build_backbone({
    'type': 'SimpleNet',
    'input_dim': 10,
    'hidden_dim': 64,
    'num_layers': 3
})

# Build head directly
head = build_head({
    'type': 'QHead',
    'in_channels': 64,
    'hidden_layers': [64, 64, 1]
})

# Build task-specific model (backbone + head composition)
model = build_network_dismantler({
    'type': 'ActorCritic',
    'backbone_cfg': {
        'type': 'SimpleNet',
        'input_dim': 10,
        'hidden_dim': 64
    },
    'actor_head_cfg': {
        'type': 'LogitHead',
        'in_channels': 64,
        'hidden_layers': [64, 64, 1]
    },
    'critic_head_cfg': {
        'type': 'VHead',
        'in_channels': 64,
        'num_critics': 2,
        'hidden_layers': [64, 64, 1]
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
| **FPNet** | Feature Pyramid Network | - Multi-scale features<br>- Feature fusion (add/concat/attention)<br>- Different depths per pyramid level |

### Heads

| Head | Description | Use Case |
|------|-------------|----------|
| **MLPHead** | Generic MLP head | Custom predictions |
| **QHead** | Q-value head | DQN, Q-learning algorithms |
| **VHead** | Value head | Critic in Actor-Critic, Advantage estimation |
| **LogitHead** | Logit head | Policy logits in Actor-Critic |
| **BranchValueHead** | Branch value head | Hierarchical value estimation with branch aggregation |

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

Feature Pyramid Network with multi-scale feature fusion.

```python
config = {
    'type': 'FPNet',
    'input_dim': 10,           # Input feature dimension
    'hidden_dims': [64, 128, 256],  # Hidden dimensions for each pyramid level
    'num_layers_list': [1, 2, 3],   # Number of GraphSAGE layers per level (optional)
    'aggr': 'mean',            # Node aggregation
    'graph_aggr': 'add',       # Graph pooling
    'norm': 'layer',           # Normalization
    'dropout': 0.1,           # Dropout
    'fusion_mode': 'attention',  # Feature fusion: 'add', 'concat', 'attention'
    'output_dim': 256          # Output dimension (default: max(hidden_dims))
}
```

## Head Configuration

### MLPHead

Generic MLP head for custom predictions.

```python
config = {
    'type': 'MLPHead',
    'in_channels': 64,         # Input feature dimension
    'hidden_layers': [64, 32, 1],  # Layer dimensions
    'activation': 'leaky_relu',    # Activation: 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'none'
    'dropout': 0.0,           # Dropout probability
    'norm': 'none'            # Normalization: 'batch', 'layer', 'none'
}
```

### QHead

Q-value prediction head for DQN.

```python
config = {
    'type': 'QHead',
    'in_channels': 64,
    'hidden_layers': [64, 64, 1],
    'activation': 'leaky_relu',
    'dropout': 0.0
}
```

### VHead

Value estimation head for critic.

```python
config = {
    'type': 'VHead',
    'in_channels': 64,
    'hidden_layers': [64, 64, 1],
    'activation': 'leaky_relu',
    'dropout': 0.0
}
```

### LogitHead

Policy logit head for Actor-Critic.

```python
config = {
    'type': 'LogitHead',
    'in_channels': 64,
    'hidden_layers': [64, 64, 1],
    'activation': 'leaky_relu',
    'dropout': 0.0
}
```

### BranchValueHead

Hierarchical value head with branch aggregation.

```python
config = {
    'type': 'BranchValueHead',
    'in_channels': 64,
    'hidden_layers': [64, 1],
    'num_critics': 2,          # Number of critics for conservative estimation
    'activation': 'leaky_relu',
    'dropout': 0.0
}
```

## Task Model Configuration

### ActorCritic

Actor-Critic model with configurable backbone, actor head, and critic heads.

```python
config = {
    'type': 'ActorCritic',
    'backbone_cfg': {
        'type': 'SimpleNet',      # or 'DeepNet', 'FPNet'
        'input_dim': 10,
        'hidden_dim': 64
    },
    'actor_head_cfg': {
        'type': 'LogitHead',
        'in_channels': 64,
        'hidden_layers': [64, 64, 1],
        'activation': 'leaky_relu'
    },
    'critic_head_cfg': {
        'type': 'VHead',
        'in_channels': 64,
        'num_critics': 2,         # Number of critic heads
        'hidden_layers': [64, 64, 1],
        'activation': 'leaky_relu'
    },
    'branch_aggr': 'add'         # Branch aggregation method
}
```

### Qnet

Q-network with configurable backbone and Q-value head.

```python
config = {
    'type': 'Qnet',
    'backbone_cfg': {
        'type': 'SimpleNet',      # or 'DeepNet', 'FPNet'
        'input_dim': 10,
        'hidden_dim': 64
    },
    'q_head_cfg': {
        'type': 'QHead',
        'in_channels': 64,
        'hidden_layers': [64, 64, 1],
        'activation': 'leaky_relu'
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
from src.models import build_backbone
backbone = build_backbone({
    'type': 'MyCustomBackbone',
    'input_dim': 10,
    'hidden_dim': 128
})
```

### Method 2: Register Custom Head

```python
import torch.nn as nn
from src.models.utils.registry import HEADS

@HEADS.register_module()
class MyCustomHead(nn.Module):
    """Custom prediction head."""

    def __init__(self, in_channels: int, hidden_layers: list = None, **kwargs):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [in_channels, 1]

        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if i < len(hidden_layers) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# Use custom head
from src.models import build_head
head = build_head({
    'type': 'MyCustomHead',
    'in_channels': 64,
    'hidden_layers': [64, 32, 1]
})
```

### Method 3: Register Custom Task Model

```python
import torch.nn as nn
from src.models.utils.registry import NETWORK_DISMANTLER
from src.models import build_backbone, build_head

@NETWORK_DISMANTLER.register_module()
class MyCustomTaskModel(nn.Module):
    """Custom task-specific model."""

    def __init__(self,
                 backbone_cfg: dict = None,
                 head_cfg: dict = None):
        super().__init__()

        self.backbone = build_backbone(backbone_cfg)
        self.head = build_head(head_cfg)

    def forward(self, x, edge_index, batch, **kwargs):
        node_embed, graph_embed = self.backbone(x, edge_index, batch)
        return self.head(node_embed)

# Use custom model
from src.models import build_network_dismantler

model = build_network_dismantler({
    'type': 'MyCustomTaskModel',
    'backbone_cfg': {
        'type': 'SimpleNet',
        'input_dim': 10,
        'hidden_dim': 64
    },
    'head_cfg': {
        'type': 'MyCustomHead',
        'in_channels': 64,
        'hidden_layers': [64, 32, 1]
    }
})
```

### Method 4: Extend Existing Backbones

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

  # Backbone configuration
  backbone_cfg:
    type: SimpleNet
    input_dim: 10
    hidden_dim: 64
    num_layers: 3
    aggr: mean
    graph_aggr: add
    norm: layer
    dropout: 0.0

  # Actor head configuration (policy head)
  actor_head_cfg:
    type: LogitHead
    in_channels: 64
    hidden_layers: [64, 64, 1]
    activation: leaky_relu

  # Critic head configuration (value head)
  critic_head_cfg:
    type: VHead
    in_channels: 64
    num_critics: 2
    hidden_layers: [64, 64, 1]
    activation: leaky_relu

  branch_aggr: add
```

### Example: Deep Network with Custom Blocks

```yaml
# configs/network_dismantling/actor_critic_deep.yaml
model:
  type: ActorCritic

  # Backbone configuration
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

  # Actor head configuration
  actor_head_cfg:
    type: LogitHead
    in_channels: 128
    hidden_layers: [128, 128, 1]
    activation: leaky_relu

  # Critic head configuration
  critic_head_cfg:
    type: VHead
    in_channels: 128
    num_critics: 3
    hidden_layers: [128, 64, 1]
    activation: leaky_relu

  branch_aggr: add
```

### Loading from YAML

```python
import yaml
from src.models import build_network_dismantler

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
print(f"Logits: {output['logit'].shape}")
print(f"Node embed: {output['node_embed'].shape}")
print(f"Graph embed: {output['graph_embed'].shape}")
```

### Example: Building Backbone and Head Separately

```python
import torch
from src.models import build_backbone, build_head

# Build backbone
backbone_cfg = {
    'type': 'SimpleNet',
    'input_dim': 10,
    'hidden_dim': 64
}
backbone = build_backbone(backbone_cfg)

# Build head
head_cfg = {
    'type': 'QHead',
    'in_channels': 64,
    'hidden_layers': [64, 64, 1]
}
head = build_head(head_cfg)

# Forward pass using info dictionary (recommended)
x = torch.randn(100, 10)
edge_index = torch.randint(0, 100, (2, 200))
batch = torch.zeros(100, dtype=torch.long)

# Create info dictionary
info = {
    'x': x,
    'edge_index': edge_index,
    'batch': batch
}

# Get features from backbone
info = backbone.forward_info(info)
print(f"Node embed shape: {info['node_embed'].shape}")
print(f"Graph embed shape: {info['graph_embed'].shape}")

# Get predictions from head
info = head.forward_info(info)
print(f"Q-values shape: {info['q_values'].shape}")
```

### Legacy Forward (Positional Arguments)

```python
# Legacy forward still works for backward compatibility
node_embed, graph_embed = backbone(x, edge_index, batch)
q_values = head(node_embed, batch, graph_embed)
```

## API Reference

### Build Functions

```python
from src.models import build_backbone, build_head, build_network_dismantler

# Build backbone
backbone = build_backbone(config_dict)

# Build head
head = build_head(config_dict)

# Build task model
model = build_network_dismantler(config_dict)
```

### Registry Access

```python
from src.models.utils.registry import BACKBONES, HEADS, NETWORK_DISMANTLER

# Get registered class
backbone_class = BACKBONES.get('SimpleNet')
head_class = HEADS.get('QHead')
model_class = NETWORK_DISMANTLER.get('ActorCritic')

# Show all registered items
print(list(BACKBONES.module_dict.keys()))        # ['SimpleNet', 'DeepNet', 'FPNet']
print(list(HEADS.module_dict.keys()))             # ['MLPHead', 'QHead', 'VHead', 'LogitHead', 'BranchValueHead']
print(list(NETWORK_DISMANTLER.module_dict.keys()))  # ['ActorCritic', 'Qnet']

# Check if registered
'SimpleNet' in BACKBONES  # True
'ActorCritic' in NETWORK_DISMANTLER  # True
```

## Best Practices

1. **Use Registry**: Always register custom models using `@BACKBONES.register_module()`, `@HEADS.register_module()`, or `@NETWORK_DISMANTLER.register_module()`
2. **Flexible Config**: Use YAML configs for easy experimentation
3. **Modular Design**: Separate backbones, heads, task models, and utility functions
4. **Documentation**: Document custom models with comprehensive docstrings
5. **Testing**: Test models with dummy data before real training
6. **Backbone + Head Composition**: Task models should compose backbones and heads for maximum flexibility
7. **Type Hints**: Use Python type hints for better IDE support
8. **Separate Imports**: Import from `src.models` for builder functions, from specific modules for classes

## Common Patterns

### Backbone with Info Dictionary

```python
@BACKBONES.register_module()
class MyBackbone(nn.Module):
    def forward(self, x, edge_index, batch):
        # Legacy forward (for backward compatibility)
        node_embed = self.encoder(x, edge_index, batch)
        graph_embed = global_mean_pool(node_embed, batch)
        return node_embed, graph_embed

    def forward_info(self, info: dict):
        # Preferred forward using info dictionary
        x = info['x']
        edge_index = info['edge_index']
        batch = info['batch']

        node_embed = self.encoder(x, edge_index, batch)
        graph_embed = global_mean_pool(node_embed, batch)

        info['node_embed'] = node_embed
        info['graph_embed'] = graph_embed
        return info
```

### Head with Info Dictionary

```python
@HEADS.register_module()
class MyHead(nn.Module):
    def forward(self, x):
        # Legacy forward
        return self.mlp(x)

    def forward_info(self, info: dict):
        # Preferred forward using info dictionary
        x = info.get('node_embed', info.get('graph_embed'))
        output = self.mlp(x)
        info['output'] = output
        return info
```

### Task Model with Backbone + Heads using Info Dictionary

```python
@NETWORK_DISMANTLER.register_module()
class MyTaskModel(nn.Module):
    def __init__(self, backbone_cfg, head1_cfg, head2_cfg):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.head1 = build_head(head1_cfg)
        self.head2 = build_head(head2_cfg)

    def forward_info(self, info: dict):
        # Get embeddings from backbone
        info = self.backbone.forward_info(info)

        # Get predictions from heads
        info = self.head1.forward_info(info)
        info = self.head2.forward_info(info)

        return info

    def forward(self, x, edge_index, batch, **kwargs):
        # Legacy forward for backward compatibility
        node_embed, graph_embed = self.backbone(x, edge_index, batch)
        output1 = self.head1(node_embed)
        output2 = self.head2(graph_embed)
        return {'output1': output1, 'output2': output2}
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to import the module before building
   ```python
   from src.models.backbones import SimpleNet  # Import to register
   from src.models.heads import QHead  # Import to register heads
   from src.models.network_dismantler import ActorCritic  # Import to register
   from src.models import build_backbone, build_head, build_network_dismantler

   backbone = build_backbone({'type': 'SimpleNet', ...})
   ```

2. **Registry Key Error**: Check if model is registered
   ```python
   from src.models.utils.registry import BACKBONES, HEADS, NETWORK_DISMANTLER

   print(list(BACKBONES.module_dict.keys()))        # Show all backbones
   print(list(HEADS.module_dict.keys()))             # Show all heads
   print(list(NETWORK_DISMANTLER.module_dict.keys()))  # Show task models
   ```

3. **Dimension Mismatch**: Ensure input_dim matches your data
   ```python
   x.shape[1] == config['input_dim']  # Should be True
   backbone.out_channels == head_cfg['in_channels']  # Should be True
   ```

4. **Missing Arguments**: Verify all required config keys are provided
   ```python
   # SimpleNet requires: type, input_dim
   # DeepNet requires: type, input_dim, hidden_dim, num_blocks
   # ActorCritic requires: type, backbone_cfg, actor_head_cfg, critic_head_cfg
   # Qnet requires: type, backbone_cfg, q_head_cfg
   ```

## Examples

See `examples/` directory for complete examples:
- `build_network_dismantler_example.py`: Building network dismantling models from configs
