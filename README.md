# CentriLearn: Learning to Identify Key Nodes in Complex Networks

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Version](https://img.shields.io/badge/version-v0.2.0-blue)](https://github.com/He-JiYe/CentriLearn/releases/tag/v0.2.0)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![中文文档](https://img.shields.io/badge/README-中文-blue.svg)](README_CN.md)

A reinforcement learning framework based on graph neural networks for solving combinatorial optimization problems in complex networks, such as **network dismantling**.

</div>

---

## Features

- **Graph-Oriented RL**: Reinforcement learning framework specifically designed for graph data based on PyTorch Geometric
- **Modular Architecture**: Clear separation of environments, algorithms, models, and metrics for easy extension to other graph combinatorial optimization problems
- **Registry System**: Flexible component registration with dynamic building, easy to modify experimental configurations
- **Rich Algorithms**: DQN and PPO reinforcement learning algorithm implementations

---

## Project Structure

```
centrilearn/
├── algorithms/          # RL algorithms (DQN, PPO)
├── buffer/              # Experience replay buffers
├── environments/        # Task environments
├── metrics/             # Evaluation metrics
├── models/              # GNN backbones & prediction heads
│   ├── backbones/       # GraphSAGE, GAT, GIN
│   └── heads/           # QHead, VHead, PolicyHead, etc.
└── utils/               # Builder, registry, training utilities
```

---

## Installation

```bash
pip install -e .
```

**Requirements**:
- Python >= 3.11
- PyTorch >= 2.7.0
- PyTorch Geometric >= 2.6.0
- torch-scatter >= 2.1.0

---

## Quick Start

### Training

```bash
# DQN training
python tools/train.py configs/network_dismantling/FINDER.yaml

# PPO training
python tools/train.py configs/network_dismantling/CentriLearn.yaml

# With custom parameters
python tools/train.py configs/network_dismantling/CentriLearn.yaml --num_episodes 500 --batch_size 64
```

### Testing

```bash
# Test trained model
python tools/test.py configs/network_dismantling/FINDER.yaml --checkpoint ./checkpoints/model.pth
```

### Configuration

All components can be configured via YAML/JSON files:

```yaml
algorithm:
  type: DQN
  lr: 0.0001
  gamma: 0.99
  epsilon_start: 1.0

model:
  backbone:
    type: GraphSAGE
    in_channels: 1
    hidden_channels: 128
    num_layers: 3
  head:
    type: QHead
    in_channels: 128
```

---

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| **DQN** | Deep Q-Network with experience replay and target network |
| **PPO** | Proximal Policy Optimization with clipped objective |

---

## Citation

If this project helps your research, please cite:

```bibtex
@misc{CentriLearn2026,
  title = {CentriLearn: A Reinforcement Learning Framework for Complex Networks},
  author = {CentriLearn Team},
  year = {2026},
  url = {https://github.com/He-JiYe/CentriLearn}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

If this project helps you, please give us a ⭐️!

</div>
