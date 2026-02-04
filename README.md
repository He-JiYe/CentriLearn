# CentriLearn: Learning to Identify Key Nodes in Complex Networks

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reinforcement learning framework based on graph neural networks for solving combinatorial optimization problems in complex networks, such as network dismantling.

</div>

---

## Update Progress

### Latest Version: v0.1.0 (Feb 2026)

#### ✅ Completed Features

**Core Framework**
- ✅ Modular architecture (separation of environments, algorithms, and models)
- ✅ Registry mechanism for dynamic component registration
- ✅ Configurable training system
- ✅ Multiple graph neural network backbones (GraphSAGE, GAT, GIN, etc.)
- ✅ Flexible prediction head system (QHead, VHead, LogitHead, etc.)

**Reinforcement Learning Algorithms**
- ✅ DQN (Deep Q-Network) implementation
- ✅ PPO (Proximal Policy Optimization) implementation
- ✅ Support for experience replay buffers (standard/prioritized)
- ✅ PPO rollout buffer (RolloutBuffer)

**Complex Network Tasks: Network Dismantling**
- ✅ Network dismantling environment (NetworkDismantlingEnv)
- ✅ Synthetic graph generation (BA, ER, etc.)
- ✅ Real-world network dataset support

#### 🚧 Future Plans

- 🔄 Support for vectorized environments
- 🔄 More reinforcement learning algorithms (A3C, SAC, TD3)
- 🔄 More application scenarios
- 🔄 More training tools
- 🔄 Distributed training support
- 🔄 Documentation improvement and performance optimization

---

## Project Motivation

### Complex Network Tasks

There are many combinatorial optimization problems in graph theory, such as network dismantling and graph partitioning, which are NP-Hard problems. Research on these problems has often relied on heuristic algorithms with handcrafted features. In recent years, an increasing number of studies have used deep reinforcement learning methods to solve these combinatorial optimization problems and achieved significant results.

### Graph Reinforcement Learning Framework

Currently, there are many mature frameworks in the fields of graph neural networks and reinforcement learning, such as PyG (PyTorch Geometric) and SB3 (Stable Baselines3), but specialized frameworks for graph reinforcement learning remain absent. Due to the uniqueness of graph data (node connections, graph structure changes, etc.), extending existing reinforcement learning frameworks poses significant challenges. Therefore, this project aims to establish a reinforcement learning framework for graph data to facilitate learning and experimentation for relevant researchers.

### Personal Motivation

I have previously conducted research on complex networks, and my thesis topic is graph reinforcement learning. Therefore, I developed this project to help me complete my thesis. This is also my first open-source project, and I hope to provide valuable tools to the community.

---

## Key Features

- **Graph Data Focused**: Reinforcement learning framework for graph data based on PyTorch Geometric
- **Modular Design**: Clear separation of environment, algorithm, and model components for easy extension and combination
- **Registry Mechanism**: Flexible component registration and dynamic building, similar to mmcv's configuration style
- **Configurable Training**: Start training with one click through configuration files without modifying code
- **Easy to Extend**: Easily register custom components through decorators, easily extendable to different complex network sequential decision-making tasks

## Documentation

- **[Modules Guide](docs/modules_guide.md)** - Comprehensive guide for using different modules (algorithms, environments, models, buffers, metrics)
- **[API Reference](docs/api_reference.md)** - Detailed API documentation for all public interfaces
- **[Examples](examples/)** - Example scripts demonstrating various use cases:
  - [DQN Example](examples/dqn_example.py) - DQN training examples
  - [PPO Example](examples/ppo_example.py) - PPO training examples

## Installation Guide

### Requirements

- Python >= 3.11
- CUDA >= 11.8 (GPU training recommended)

### Quick Installation

#### Method 1: Install with pip

```bash
# Clone the project
git clone https://github.com/He-JiYe/CentriLearn.git
cd CentriLearn

# Install core dependencies
pip install -e .

# Install all dependencies (recommended)
pip install -e ".[all]"
```

#### Method 2: Manual Installation

```bash
# Install PyTorch (select according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other dependencies
pip install networkx numpy pyyaml tqdm
```

## Quick Start

> ⚠️ **Note**: This project is still under active development. The following content provides basic usage examples. More detailed documentation, tutorials, and API references will be provided in the future.

### Method 1: Command Line Training (Recommended)

We provide a convenient command-line tool to start training directly through YAML configuration files:

```bash
# Basic training
python tools/train.py configs/network_dismantling/dqn.yaml

# Enable logging
python tools/train.py configs/network_dismantling/dqn.yaml --use_logging --log_dir ./logs/train

# Specify checkpoint save directory
python tools/train.py configs/network_dismantling/dqn.yaml --ckpt_dir ./checkpoints

# Resume training from checkpoint
python tools/train.py configs/network_dismantling/dqn.yaml --resume ./checkpoints/checkpoint_episode_500.pth

# Customize training parameters
python tools/train.py configs/network_dismantling/ppo.yaml --num_episodes 500 --batch_size 64 --save_interval 50
```

### Method 2: Python Code Training

```python
import yaml
from src.utils import train_from_cfg

# Load configuration file
with open('configs/network_dismantling/dqn.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Start training
results, algorithm = train_from_cfg(config, verbose=True)

# Access training results
print(f"Average reward: {results['avg_reward']:.4f}")
print(f"Total episodes: {results['total_episodes']}")
```

### Method 3: Custom Training Flow

```python
import networkx as nx
from src.utils import build_environment, build_algorithm

# Create custom environment
graph = nx.barabasi_albert_graph(n=50, m=2)
env = build_environment({
    'type': 'NetworkDismantlingEnv',
    'graph': graph,
    'node_features': 'combin'
})

# Build algorithm
algo = build_algorithm({
    'type': 'DQN',
    'model': {
        'type': 'Qnet',
        'backbone_cfg': {
            'type': 'GraphSAGE',
            'in_channels': 2,
            'hidden_channels': 64,
            'num_layers': 3
        },
        'q_head_cfg': {
            'type': 'QHead',
            'in_channels': 64
        }
    },
    'optimizer_cfg': {
        'type': 'Adam',
        'lr': 0.0001
    },
    'algo_cfg': {
        'gamma': 0.99,
        'epsilon_decay': 10000
    },
    'device': 'cuda'
})

# Train
results = algo._run_training_loop(env, {
    'num_episodes': 1000,
    'batch_size': 32,
    'log_interval': 10,
    'ckpt_dir': './checkpoints',
    'save_interval': 100
})
```

---

## Configuration File

CentriLearn uses YAML/JSON format configuration files with highly flexible configuration. Specific parameters match the model requirements and can be determined by checking the model code. Below is a sample YAML configuration file:

```yaml
algorithm:
  type: DQN                              # Algorithm type: DQN | PPO
  model:
    type: Qnet                            # Model type
    backbone_cfg:                          # Backbone network config
      type: GraphSAGE                     # Supports multiple GNNs
      in_channels: 2
      hidden_channels: 64
      num_layers: 3
    q_head_cfg:                           # Q-value prediction head
      type: QHead
      in_channels: 64
  optimizer_cfg:                          # Optimizer config
    type: Adam
    lr: 0.0001
    weight_decay: 0.0005
  replaybuffer_cfg:                       # Experience replay buffer
    type: PrioritizedReplayBuffer
    capacity: 10000
  metric_manager_cfg:                     # Metric manager
    save_dir: ./logs/metrics
    log_interval: 10
    metrics:
      - type: AUC                         # Giant connected component area
        record: min
      - type: AttackRate                  # Attack rate
        record: min
  algo_cfg:                               # Algorithm hyperparameters
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 10000
    tau: 0.005
  device: cuda

environment:
  type: NetworkDismantlingEnv              # Environment type
  synth_type: ba                          # Synthetic graph type
  synth_args:
    min_n: 30
    max_n: 50
    m: 4
  node_features: combin                    # Node feature type
  is_undirected: True
  value_type: ar                          # Reward type: ar (attack rate)
  use_gcc: False
  use_component: False
  device: cuda

training:
  num_episodes: 1000                      # Number of training episodes
  max_steps: 1000                         # Max steps per episode
  batch_size: 32                          # Batch size
  log_interval: 10                         # Log print interval
  eval_interval: 100                       # Evaluation interval
  eval_episodes: 5                         # Number of evaluation episodes
  ckpt_dir: ./checkpoints                 # Checkpoint save directory
  save_interval: 100                      # Checkpoint save interval
  resume: null                             # Resume path
```

### Supported Components

#### Algorithms
- `DQN`: Deep Q-Network
- `PPO`: Proximal Policy Optimization

#### Backbone Networks
- `GraphSAGE`: Graph SAGE
- `GAT`: Graph Attention Network
- `GIN`: Graph Isomorphism Network
- `DeepNet`: Deep Graph Neural Network
- `FPNet`: Feature Pyramid Graph Neural Network

#### Prediction Heads
- `QHead`: Q-value prediction head
- `VHead`: Value prediction head
- `LogitHead`: Policy prediction head
- `PolicyHead`: Policy head

#### Environment Types
- `NetworkDismantlingEnv`: Network dismantling environment
- `VectorizedEnv`: Vectorized environment (parallel training)

#### Buffers
- `ReplayBuffer`: Standard experience replay
- `PrioritizedReplayBuffer`: Prioritized experience replay
- `RolloutBuffer`: PPO rollout buffer

---

## Advanced Features

### Vectorized Environment Training (Under Development)

Using vectorized environments can significantly improve training efficiency by running multiple environment instances simultaneously:

```python
from src.environments import VectorizedEnv

# Create vectorized environment
env = VectorizedEnv({
    'env_kwargs': {
        'type': 'NetworkDismantlingEnv',
        'synth_type': 'ba',
        'synth_args': {'min_n': 30, 'max_n': 50, 'm': 4},
        # ...
    },
    'env_num': 4  # 4 parallel environments
})

# Training automatically detects and uses vectorized mode
results = algo._run_training_loop(env, training_cfg)
```

Or in configuration file:

```yaml
environment:
  type: VectorizedEnv
  env_kwargs:
    type: NetworkDismantlingEnv
    synth_type: ba
    # ...
  env_num: 4
```

### Checkpoint Recovery

Checkpoints are automatically saved during training and can be resumed from:

```bash
# Automatically save during training
python tools/train.py configs/dqn.yaml --ckpt_dir ./checkpoints

# Resume after interruption
python tools/train.py configs/dqn.yaml --resume ./checkpoints/checkpoint_episode_500.pth
```

Saved checkpoints include:
- Model parameters (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Learning rate scheduler state (`scheduler_state_dict`)
- Training steps (`training_step`)
- Training progress and statistics

### Metric Recording and Evaluation

Built-in multiple evaluation metrics automatically record the training process:

```yaml
metric_manager_cfg:
  save_dir: ./logs/metrics
  log_interval: 10
  metrics:
    - type: AUC           # Area under giant connected component curve
      record: min
    - type: AttackRate    # Attack rate
      record: min
    - type: EpisodeReward # Cumulative reward
      record: max
```

Metric history is automatically saved as JSON files for subsequent analysis.

---

## Project Structure

```
CentriLearn/
├── configs/                    # Configuration files
│   └── network_dismantling/    # Network dismantling configs
│       ├── dqn.yaml
│       ├── ppo.yaml
│       └── dqn_vectorized.yaml
├── ckpt/                       # Model weights
├── data/                       # Datasets
│   ├── small/                  # Small-scale networks
│   └── large/                  # Large-scale networks
├── docs/                       # Documentation
├── logs/                       # Logs
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
│   ├── algorithms/             # RL algorithms
│   │   ├── base.py            # Algorithm base class
│   │   ├── dqn.py             # DQN implementation
│   │   └── ppo.py             # PPO implementation
│   ├── buffer/                 # Experience buffers
│   │   ├── base.py
│   │   ├── replaybuffer.py
│   │   └── rolloutbuffer.py
│   ├── environments/           # Environment implementations
│   │   ├── base.py
│   │   ├── network_dismantling.py
│   │   └── vectorized_env.py
│   ├── metrics/               # Evaluation metrics
│   │   ├── base.py
│   │   ├── manager.py
│   │   └── network_dismantling_metrics.py
│   ├── models/                # Model components
│   │   ├── backbones/         # Backbone networks
│   │   │   ├── GraphSAGE.py
│   │   │   ├── GAT.py
│   │   │   ├── GIN.py
│   │   │   ├── DeepNet.py
│   │   │   └── FPNet.py
│   │   ├── heads/             # Prediction heads
│   │   │   ├── q_head.py
│   │   │   ├── v_head.py
│   │   │   ├── logit_head.py
│   │   │   └── policy_head.py
│   │   ├── network_dismantler/ # Complete models
│   │   │   ├── Qnet.py
│   │   │   └── ActorCritic.py
│   │   └── loss/              # Loss functions
│   │       └── restruct_loss.py
│   └── utils/                 # Utilities
│       ├── builder.py          # Component builder
│       ├── registry.py        # Registry
│       └── train.py           # Training entry
├── tests/                     # Tests
├── tools/                     # Tools
│   └── train.py              # Training script
├── pyproject.toml            # Project configuration
├── README.md                 # English documentation
└── README_CN.md             # Chinese documentation
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Submit a Pull Request

### Code Standards

- Format code with Black: `black src/`
- Sort imports with isort: `isort src/`
- Run tests: `pytest`
- Check types: `mypy src/`

---

## FAQ

### Q1: How to use my own network data?

**A:** You can load real-world network data and create an environment:

```python
import networkx as nx
from src.utils import build_environment

# Load network data
graph = nx.read_edgelist('data/my_network.edgelist')

# Create environment
env = build_environment({
    'type': 'NetworkDismantlingEnv',
    'graph': graph,
    'node_features': 'combin'
})
```

### Q2: What if training is slow?

**A:** Try the following methods to improve training speed:
1. Use vectorized environments for parallel training
2. Increase `batch_size`
3. Use GPU training (`device: cuda`)
4. Reduce model complexity

We will further optimize project performance in the future.

### Q3: How to add custom algorithms?

**A:** Use the registry decorator to register your algorithm:

```python
from src.utils import ALGORITHMS

@ALGORITHMS.register_module()
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, ...):
        # Implement your algorithm
        pass
```

Then use it in the configuration file:
```yaml
algorithm:
  type: MyAlgorithm
  # ...
```

### Q4: How to evaluate a trained model?

**A:** Load a checkpoint and evaluate on the test set:

```python
from src.utils import build_algorithm

# Build algorithm
algo = build_algorithm(algorithm_cfg)

# Load checkpoint
algo.load_checkpoint('checkpoints/model_best.pth')

# Set to evaluation mode
algo.set_eval_mode()

# Evaluate in test environment
# ...
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- Project Homepage: [https://github.com/He-JiYe/CentriLearn](https://github.com/He-JiYe/CentriLearn)
- Issue Reporting: [Issues](https://github.com/He-JiYe/CentriLearn/issues)
- Email: 202200820169@mail.sdu.edu.cn

---

<div align="center">

If this project helps you, please give us a ⭐️!

</div>
