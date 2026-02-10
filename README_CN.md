# CentriLearn: 学习识别复杂网络中的核心节点

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-orange.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-v0.2.0-blue)](https://github.com/He-JiYe/CentriLearn/releases/tag/v0.2.0)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![英文文档](https://img.shields.io/badge/README-English-blue.svg)](README.md)

一个基于图神经网络的强化学习框架，用于解决复杂网络中的组合优化问题，如网络瓦解等。

</div>

---

## 更新进度

### 最新版本：v0.2.0 (2026.02.10)

#### ✅ 已完成功能

**核心框架**
- ✅ 模块化架构设计（环境、算法、模型分离）
- ✅ 注册器机制（Registry）实现动态组件注册
- ✅ 配置化训练系统
- ✅ 多种图神经网络骨干网络（GraphSAGE、GAT、GIN 等）
- ✅ 灵活的预测头系统（QHead、VHead、LogitHead 等）

**强化学习算法**
- ✅ DQN (Deep Q-Network) 实现
- ✅ PPO (Proximal Policy Optimization) 实现
- ✅ 支持经验回放缓冲区（标准/优先级）
- ✅ PPO 轨迹缓冲区（RolloutBuffer）

**复杂网络任务: 网络瓦解**
- ✅ 网络瓦解环境（NetworkDismantlingEnv）
- ✅ 合成图生成（BA、ER 等）
- ✅ 真实网络数据集支持


#### ✅ 更新日志（2026.02.10）
**Bug 修复**
- ✅ 修复 algorithms/backbones/ 中 GAT、GIN 的 graph_embed 未定义
- ✅ 修复 DQN 算法中的 double dqn 重复计算以及错误使用 argmax 方法

**新增功能**
- ✅ 定时保存功能: 为DQN和PPO算法添加定期保存模型检查点
- ✅ 恢复训练功能: 为训练流程添加resume功能
- ✅ 支持多线程向量化环境（VectorizedEnv）训练
  
**性能优化**
- ✅ 优化连通分量计算性能（递归 → 迭代实现）
- ✅ 优化 DQN 训练过程（梯度裁剪 + 目标网络更新频率调整）
- ✅ 优化训练性能和内存效率

#### 🚧 未来计划

- 🔄 更多强化学习算法（A3C、SAC、TD3）
- 🔄 更多应用场景
- 🔄 更多训练工具
- 🔄 分布式训练支持
- 🔄 文档完善和性能优化
- 🔄 大规模测试和评估
- 🔄 用 Rust 重写核心模块，提升训练效率


---

## 项目动机

### 复杂网络任务

图论中有许多组合优化问题，例如网络瓦解、图分割等，这些任务都是 NP-Hard 问题。过去这些问题的研究往往依赖于手工设计特征的启发式算法。近年来，越来越多的研究通过深度强化学习方法来解决这些组合优化问题，并取得了显著的成果。

### 图强化学习框架

目前在图神经网络和强化学习领域有许多成熟的框架，例如 PyG（PyTorch Geometric）、SB3（Stable Baselines3）等，但关于图强化学习的专门框架仍然处于空白。由于图数据的特殊性（节点连接关系、图结构变化等），在已有的强化学习框架上进行扩展具有较大挑战。因此，本项目希望建立一个针对图数据的强化学习框架，便于相关研究者进行学习和实验。

### 个人动机

由于我过去从事复杂网络相关研究，并且我的毕业论文选题为图强化学习，因此开发这个项目来帮助我顺利完成毕业课题。同时，这也是我开发的第一个开源项目，希望能够为社区提供有价值的工具。

---


### 核心特性

- **针对图数据类型**: 基于 PyTorch Geometric 实现的针对图数据的强化学习框架
- **模块化设计**: 清晰分离环境、算法、模型组件，便于扩展和组合使用
- **注册器机制**: 灵活的组件注册和动态构建，类似 mmcv 的配置化风格
- **配置化训练**: 通过配置文件一键启动训练，无需修改代码
- **易于扩展**: 通过装饰器轻松注册自定义组件，易于扩展到不同的复杂网络序列决策任务

## 文档

- **[模块指南](docs/modules_guide.md)** - 模块使用指南（算法、环境、模型、缓冲区、指标）
- **[API 参考文档](docs/api_reference.md)** - 所有公共接口的详细 API 文档
- **[示例代码](examples/)** - 各种使用场景的示例脚本：
  - [DQN 示例](examples/dqn_example.py) - DQN 训练示例
  - [PPO 示例](examples/ppo_example.py) - PPO 训练示例

## 安装指南

### 环境要求

- Python >= 3.11
- CUDA >= 11.8 (推荐使用 GPU 训练)

### 快速安装

#### 方式一：使用 pip 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/CentriLearn.git
cd CentriLearn

# 安装依赖
pip install -e .
```

#### 方式二：手动安装依赖

```bash
# 安装 PyTorch (根据您的 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

# 安装其他依赖
pip install networkx numpy pyyaml tqdm
```

## 快速开始

> ⚠️ **注意**: 项目目前仍在积极开发中，以下内容为基本使用示例。未来将提供更详细的使用文档、教程和 API 参考。

### 方式一：命令行训练（推荐）

我们提供了便捷的命令行工具，可以直接通过 YAML 配置文件启动训练：

```bash
# 基本训练
python tools/train.py configs/network_dismantling/dqn.yaml

# 启用日志记录
python tools/train.py configs/network_dismantling/dqn.yaml --use_logging --log_dir ./logs/train

# 指定 checkpoint 保存目录
python tools/train.py configs/network_dismantling/dqn.yaml --ckpt_dir ./checkpoints

# 从 checkpoint 恢复训练
python tools/train.py configs/network_dismantling/dqn.yaml --resume ./checkpoints/checkpoint_episode_500.pth

# 自定义训练参数
python tools/train.py configs/network_dismantling/ppo.yaml --num_episodes 500 --batch_size 64 --save_interval 50
```

### 方式二：Python 代码训练

```python
import yaml
from centrilearn.utils import train_from_cfg

# 加载配置文件
with open('configs/network_dismantling/dqn.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 开始训练
results, algorithm = train_from_cfg(config, verbose=True)

# 访问训练结果
print(f"平均奖励: {results['avg_reward']:.4f}")
print(f"训练轮数: {results['total_episodes']}")
```

### 方式三：自定义训练流程

```python
import networkx as nx
from centrilearn.utils import build_environment, build_algorithm

# 创建自定义环境
graph = nx.barabasi_albert_graph(n=50, m=2)
env = build_environment({
    'type': 'NetworkDismantlingEnv',
    'graph': graph,
    'node_features': 'combin',
    'is_undirected': True
})

# 构建算法
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

# 训练
results = algo._run_training_loop(env, {
    'num_episodes': 1000,
    'batch_size': 32,
    'log_interval': 10,
    'ckpt_dir': './checkpoints',
    'save_interval': 100
})
```

---

## 配置文件说明

CentriLearn 使用 YAML/JSON 格式的配置文件，支持高度灵活的配置。具体的参数与模型所需参数相符，可通过查看模型代码确定设置哪些参数。以下是一个 YAML 格式的配置文件示例：

```yaml
algorithm:
  type: DQN                              # 算法类型: DQN | PPO
  model:
    type: Qnet                            # 模型类型
    backbone_cfg:                          # 骨干网络配置
      type: GraphSAGE                     # 支持多种 GNN
      in_channels: 2
      hidden_channels: 64
      num_layers: 3
    q_head_cfg:                           # Q值预测头
      type: QHead
      in_channels: 64
  optimizer_cfg:                          # 优化器配置
    type: Adam
    lr: 0.0001
    weight_decay: 0.0005
  replaybuffer_cfg:                       # 经验回放缓冲区
    type: PrioritizedReplayBuffer
    capacity: 10000
  metric_manager_cfg:                     # 指标管理器
    save_dir: ./logs/metrics
    log_interval: 10
    metrics:
      - type: AUC                         # 攻击曲线下面积
        record: min
      - type: AttackRate                  # 攻击率
        record: min
  algo_cfg:                               # 算法超参数
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 10000
    tau: 0.005
  device: cuda

environment:
  type: NetworkDismantlingEnv              # 环境类型
  synth_type: ba                          # 合成图类型
  synth_args:
    min_n: 30
    max_n: 50
    m: 4
  node_features: combin                    # 节点特征类型
  env_num: 1                              # 环境并行数( >1 时启动向量化环境训练)            
  is_undirected: True
  value_type: ar                          # 奖励类型: ar (attack rate)
  use_gcc: False
  use_component: False
  device: cuda

training:
  num_episodes: 1000                      # 训练轮数
  max_steps: 1000                         # 每轮最大步数
  batch_size: 32                          # 批次大小
  log_interval: 10                         # 日志打印间隔
  eval_interval: 100                       # 评估间隔
  eval_episodes: 5                         # 评估轮数
  ckpt_dir: ./checkpoints                 # checkpoint 保存目录
  save_interval: 100                      # checkpoint 保存间隔
  resume: null                             # 断点恢复路径
```

### 支持的组件

#### 算法
- `DQN`: Deep Q-Network
- `PPO`: Proximal Policy Optimization

#### 骨干网络
- `GraphSAGE`: Graph SAGE
- `GAT`: Graph Attention Network
- `GIN`: Graph Isomorphism Network
- `DeepNet`: Deep Graph Neural Network
- `FPNet`: Feature Pyramid Graph Neural Network

#### 预测头
- `QHead`: Q值预测头
- `VHead`: 价值预测头
- `LogitHead`: 策略预测头
- `PolicyHead`: 策略头

#### 环境类型
- `NetworkDismantlingEnv`: 网络瓦解环境
- `VectorizedEnv`: 向量化环境（并行训练）

#### 缓冲区
- `ReplayBuffer`: 标准经验回放（支持优先级采样和N-step采样）
- `RolloutBuffer`: PPO 轨迹缓冲区

---

## 高级功能

### 向量化环境训练

使用向量化环境可以大幅提升训练效率，支持同时运行多个环境实例：

```python
from centrilearn.environments import VectorizedEnv

# 创建向量化环境
env = VectorizedEnv({
    'env_kwargs': {
        'type': 'NetworkDismantlingEnv',
        'synth_type': 'ba',
        'synth_args': {'min_n': 30, 'max_n': 50, 'm': 4},
        # ...
    },
    'env_num': 4  # 4个并行环境
})

# 训练会自动检测并使用向量化模式
results = algo._run_training_loop(env, training_cfg)
```

或在配置文件中：

```yaml
environment:
  type: VectorizedEnv
  env_kwargs:
    type: NetworkDismantlingEnv
    synth_type: ba
    # ...
  env_num: 4
```

### 断点恢复

训练过程中会自动保存 checkpoint，支持从断点恢复：

```bash
# 训练时自动保存
python tools/train.py configs/dqn.yaml --ckpt_dir ./checkpoints

# 中断后恢复训练
python tools/train.py configs/dqn.yaml --resume ./checkpoints/checkpoint_episode_500.pth
```

保存的 checkpoint 包含：
- 模型参数 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 学习率调度器状态 (`scheduler_state_dict`)
- 训练步数 (`training_step`)
- 训练进度和统计数据

### 指标记录与评估

内置多种评估指标，自动记录训练过程：

```yaml
metric_manager_cfg:
  save_dir: ./logs/metrics
  log_interval: 10
  metrics:
    - type: AUC           # 最大连通分量面积曲线下面积
      record: min
    - type: AttackRate    # 攻击率
      record: min
```

指标历史会自动保存为 JSON 文件，方便后续分析。

---

## 项目结构

```
CentriLearn/
├── configs/                    # 配置文件目录
│   └── network_dismantling/    # 网络瓦解任务配置
│       ├── dqn.yaml
│       ├── ppo.yaml
│       ├── dqn_vectorized.yaml
│       └── ppo_vectorized.yaml
├── checkpoints/                # 模型权重保存目录
├── data/                       # 数据集目录
│   ├── small/                  # 小规模网络
│   └── large/                  # 大规模网络
├── docs/                       # 文档目录
├── logs/                       # 日志目录
├── centrilearn/                # 源代码目录
│   ├── algorithms/             # 强化学习算法
│   │   ├── base.py             # 算法基类
│   │   ├── dqn.py              # DQN 实现
│   │   └── ppo.py              # PPO 实现
│   ├── buffer/                 # 经验缓冲区
│   │   ├── base.py
│   │   ├── replaybuffer.py
│   │   └── rolloutbuffer.py
│   ├── environments/           # 环境实现
│   │   ├── base.py
│   │   ├── network_dismantling.py
│   │   └── vectorized_env.py
│   ├── metrics/                # 评估指标
│   │   ├── base.py
│   │   ├── manager.py
│   │   └── network_dismantling_metrics.py
│   ├── models/                 # 模型组件
│   │   ├── backbones/          # 骨干网络
│   │   │   ├── GraphSAGE.py
│   │   │   ├── GAT.py
│   │   │   ├── GIN.py
│   │   │   ├── DeepNet.py
│   │   │   └── FPNet.py
│   │   ├── heads/              # 预测头
│   │   │   ├── q_head.py
│   │   │   ├── v_head.py
│   │   │   ├── logit_head.py
│   │   │   └── policy_head.py
│   │   ├── network_dismantler/ # 完整模型
│   │   │   ├── Qnet.py
│   │   │   └── ActorCritic.py
│   │   └── loss/               # 损失函数
│   │       └── restruct_loss.py
│   └── utils/                  # 工具模块
│       ├── builder.py          # 组件构建器
│       ├── registry.py         # 注册器
│       └── train.py            # 训练入口
├── tools/                      # 工具脚本
│   └── train.py                # 训练脚本
├── pyproject.toml              # 项目配置
├── README.md                   # 英文说明
└── README_CN.md                # 中文说明（本文件）
```

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 代码规范

- 使用 Black 格式化代码: `black centrilearn/`
- 使用 isort 排序导入: `isort centrilearn/`
- 运行测试: `pytest`
- 检查类型: `mypy centrilearn/`

---

## 常见问题

### Q1: 如何使用自己的网络数据？

**A:** 您可以加载真实网络数据，然后创建环境：

```python
import networkx as nx
from centrilearn.utils import build_environment

# 加载网络数据
graph = nx.read_edgelist('data/my_network.edgelist')

# 创建环境
env = build_environment({
    'type': 'NetworkDismantlingEnv',
    'graph': graph,
    'node_features': 'combin'
})
```

### Q2: 训练速度慢怎么办？

**A:** 可以尝试以下方法提升训练速度：
1. 使用向量化环境进行并行训练
2. 增加 `batch_size`
3. 使用 GPU 训练 (`device: cuda`)
4. 减小模型的复杂度
未来，我们会对项目的性能进行进一步优化。

### Q3: 如何添加自定义算法？

**A:** 使用注册器装饰器注册您的算法：

```python
from centrilearn.utils import ALGORITHMS

@ALGORITHMS.register_module()
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, ...):
        # 实现您的算法
        pass
```

然后在配置文件中使用：
```yaml
algorithm:
  type: MyAlgorithm
  # ...
```

### Q4: 如何评估训练好的模型？

**A:** 加载 checkpoint 并在测试集上评估：

```python
from centrilearn.utils import build_algorithm

# 构建算法
algo = build_algorithm(algorithm_cfg)

# 加载 checkpoint
algo.load_checkpoint('checkpoints/model_best.pth')

# 设置为评估模式
algo.set_eval_mode()

# 在测试环境中评估
# ...
```

---


## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- 项目主页: [https://github.com/He-JiYe/CentriLearn](https://github.com/He-JiYe/CentriLearn)
- 问题反馈: [Issues](https://github.com/He-JiYe/CentriLearn/issues)
- 邮箱: 202200820169@mail.sdu.edu.cn

---

<div align="center">

如果这个项目对您有帮助，请给我们一个 ⭐️！

</div>
