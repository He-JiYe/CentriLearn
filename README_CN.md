# CentriLearn: 学习识别复杂网络中的核心节点

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![英文文档](https://img.shields.io/badge/README-English-blue.svg)](README.md)

这是一个基于图神经网络的强化学习框架，用于解决复杂网络中的组合优化问题，如**网络瓦解**等。

</div>

---

## 核心特性

- **图数据强化学习**: 基于 PyTorch Geometric 实现的图数据专用强化学习框架
- **模块化架构**: 环境、算法、模型和指标系统，便于扩展其他图论组合优化问题
- **注册器机制**: 灵活的组件注册和动态构建，便于修改实验配置
- **强化学习算法**: 实现了 DQN 和 PPO 强化学习算法

---

## 项目结构

```
centrilearn/
├── algorithms/          # 强化学习算法 (DQN, PPO)
├── buffer/              # 经验回放缓冲区
├── environments/        # 任务环境
├── metrics/             # 评估指标
├── models/              # 图神经网络骨干与预测头
│   ├── backbones/       # GraphSAGE, GAT, GIN
│   └── heads/           # QHead, VHead, PolicyHead 等
└── utils/               # 构建器、注册器、训练工具
```

---

## 安装

```bash
pip install -e .
```

**依赖要求**:
- Python >= 3.11
- PyTorch >= 2.7.0
- PyTorch Geometric >= 2.6.0
- torch-scatter >= 2.1.0

---

## 快速开始

### 训练

```bash
# DQN 训练
python tools/train.py configs/network_dismantling/FINDER.yaml

# PPO 训练
python tools/train.py configs/network_dismantling/CentriLearn.yaml

# 自定义参数
python tools/train.py configs/network_dismantling/CentriLearn.yaml --num_episodes 500 --batch_size 64
```

### 测试

```bash
# 测试训练好的模型
python tools/test.py configs/network_dismantling/FINDER.yaml --checkpoint ./checkpoints/model.pth
```

### 配置说明

所有组件都可通过 YAML/JSON 文件配置：

```yaml
algorithm:
  type: DQN
  model_cfg:
    type: Qnet
    backbone_cfg:
      type: GraphSAGE
      in_channels: 1
      hidden_channels: 64
      num_layers: 3
    q_head_cfg:
      type: QHead
      in_channels: 128
  device: cuda
```

---

## 支持的算法

| 算法 | 描述 |
|------|------|
| **DQN** | 深度 Q 网络，可使用优先经验回放 |
| **PPO** | 近端策略优化，使用裁剪目标函数 |

---

## 引用

如果该项目对您的研究有帮助，请引用：

```bibtex
@misc{CentriLearn2026,
  title = {CentriLearn: A Reinforcement Learning Framework for Complex Networks},
  author = {CentriLearn Team},
  year = {2026},
  url = {https://github.com/He-JiYe/CentriLearn}
}
```

---

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

如果这个项目对您有帮助，请给我们一个 ⭐️！

</div>
