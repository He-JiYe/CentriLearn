# CentriLearn 模块指南

本文档介绍 CentriLearn 各模块的使用方法和最佳实践。

## 目录

- [算法模块 (algorithms)](#算法模块-algorithms)
- [环境模块 (environments)](#环境模块-environments)
- [模型模块 (models)](#模型模块-models)
- [缓冲区模块 (buffer)](#缓冲区模块-buffer)
- [指标模块 (metrics)](#指标模块-metrics)
- [工具模块 (utils)](#工具模块-utils)

---

## 算法模块 (algorithms)

### 概述

算法模块实现了强化学习算法，包括 DQN 和 PPO。所有算法继承自 `BaseAlgorithm` 基类。

### BaseAlgorithm 基类

`BaseAlgorithm` 是所有强化学习算法的基类，定义了统一的接口。

#### 初始化

```python
from centrilearn.algorithms import BaseAlgorithm

algorithm = BaseAlgorithm(
    model=model,                      # 模型实例
    optimizer_cfg={...},              # 优化器配置
    scheduler_cfg=None,               # 学习率调度器配置（可选）
    replaybuffer_cfg=None,            # 经验缓冲区配置（可选）
    metric_manager_cfg=None,          # 指标管理器配置（可选）
    device='cpu'                      # 运行设备
)
```

#### 常用方法

```python
# 设置训练/评估模式
algorithm.set_train_mode()
algorithm.set_eval_mode()

# 保存和加载检查点
algorithm.save_checkpoint('path/to/checkpoint.pth', episode=100)
algorithm.load_checkpoint('path/to/checkpoint.pth')

# 获取当前学习率
lr = algorithm.get_lr()

# 获取模型
model = algorithm.get_model()
```

#### 抽象方法（子类必须实现）

- `_build_model(model_cfg)` - 从配置构建模型
- `train_step(batch)` - 执行一步训练
- `update(*args, **kwargs)` - 更新模型参数
- `select_action(state, **kwargs)` - 选择动作
- `collect_experience(state, *args, **kwargs)` - 收集经验

### DQN (Deep Q-Network)

DQN 是基于价值的强化学习算法，使用 Q-network 学习状态-动作价值函数。

#### 初始化

```python
from centrilearn.utils import build_algorithm

algo_cfg = {
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
        'lr': 0.0001,
        'weight_decay': 0.0005
    },
    'replaybuffer_cfg': {
        'type': 'ReplayBuffer',
        'capacity': 10000,
        'n_step': 5
    },
    'algo_cfg': {
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 10000,
        'tau': 0.005
    },
    'device': 'cuda'
}

dqn = build_algorithm(algo_cfg)
```

#### DQN 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gamma` | float | 0.99 | 折扣因子 |
| `epsilon_start` | float | 1.0 | 初始探索率 |
| `epsilon_end` | float | 0.01 | 最终探索率 |
| `epsilon_decay` | int | 10000 | 探索率衰减步数 |
| `tau` | float | 0.005 | 软更新系数 |
| `rcst_coef` | float | 0.0001 | 重建损失系数 |

#### 使用示例

```python
# 选择动作（epsilon-greedy策略）
state = env.reset()
action, epsilon = dqn.select_action(state, epsilon=None)

# 收集经验到缓冲区
dqn.collect_experience(state, action, reward, next_state, done)

# 更新模型（从缓冲区采样）
loss_info = dqn.update(batch_size=32)

# 获取 Q 值
q_values = dqn.get_q_values(state)
```

#### 训练流程

```python
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # 选择动作
        action, epsilon = dqn.select_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)

        # 收集经验
        dqn.collect_experience(state, action, reward, next_state, done)

        # 更新模型
        if len(dqn.replay_buffer) > batch_size:
            loss_info = dqn.update(batch_size)

        state = next_state
        episode_reward += reward

        if done:
            break
```

### PPO (Proximal Policy Optimization)

PPO 是基于策略的强化学习算法，使用 Actor-Critic 架构。

#### 初始化

```python
algo_cfg = {
    'type': 'PPO',
    'model': {
        'type': 'ActorCritic',
        'backbone_cfg': {
            'type': 'GraphSAGE',
            'in_channels': 2,
            'hidden_channels': 64,
            'num_layers': 3
        },
        'actor_head_cfg': {
            'type': 'PolicyHead',
            'in_channels': 64
        },
        'critic_head_cfg': {
            'type': 'VHead',
            'in_channels': 64
        }
    },
    'optimizer_cfg': {
        'type': 'Adam',
        'lr': 0.0001
    },
    'replaybuffer_cfg': {
        'type': 'RolloutBuffer',
        'capacity': 2048
    },
    'algo_cfg': {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'num_epochs': 10
    },
    'device': 'cuda'
}

ppo = build_algorithm(algo_cfg)
```

#### PPO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gamma` | float | 0.99 | 折扣因子 |
| `gae_lambda` | float | 0.95 | GAE lambda参数 |
| `clip_epsilon` | float | 0.2 | PPO裁剪参数 |
| `entropy_coef` | float | 0.01 | 熵正则化系数 |
| `value_coef` | float | 0.5 | 价值损失系数 |
| `max_grad_norm` | float | 0.5 | 最大梯度裁剪 |
| `num_epochs` | int | 1 | 每次更新的epoch数 |

#### 使用示例

```python
# 选择动作
state = env.reset()
action, log_prob, value = ppo.select_action(state, deterministic=False)

# 收集经验
ppo.collect_experience(state, action, log_prob, reward, done, value)

# 更新模型（当缓冲区满时）
if len(ppo.replay_buffer) >= batch_size:
    loss_info = ppo.update(batch_size)

# 推理模式（仅获取动作和价值）
action, value = ppo.get_action_value(state)
```

#### 训练流程

```python
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # 选择动作
        action, log_prob, value = ppo.select_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)

        # 收集经验
        ppo.collect_experience(state, action, log_prob, reward, done, value)

        state = next_state
        episode_reward += reward

        if done:
            break

    # Episode 结束后更新
    if len(ppo.replay_buffer) > 0:
        loss_info = ppo.update(batch_size=64)
```

---

## 环境模块 (environments)

### 概述

环境模块定义了强化学习环境的接口和具体实现。

### BaseEnv 基类

`BaseEnv` 是所有环境的基类，定义了环境的基本接口。

#### 初始化

```python
from centrilearn.environments import BaseEnv

env = BaseEnv(
    graph=None,                     # networkx.Graph 实例
    synth_type='ba',                # 合成图类型: 'ba', 'er', 'ws'
    synth_args=None,                # 合成图参数
    node_features='ones',           # 节点特征类型: 'ones', 'degree', 'combin'
    use_component=False,            # 是否使用连通分量
    is_undirected=True,             # 是否无向图
    device='cpu'                    # 计算设备
)
```

#### 常用方法

```python
# 重置环境
state = env.reset(graph=optional_graph)

# 获取当前状态
state = env.get_state()

# 获取 PyG 格式数据
pyg_data = env.get_pyg_data(mask=None)

# 检查图是否为空
is_empty = env.is_empty()
```

### NetworkDismantlingEnv

网络瓦解环境，用于评估在网络中移除节点的效果。

#### 初始化

```python
from centrilearn.utils import build_environment

# 使用真实图
import networkx as nx
graph = nx.karate_club_graph()

env_cfg = {
    'type': 'NetworkDismantlingEnv',
    'graph': graph,
    'node_features': 'combin',
    'value_type': 'auc',          # 'auc' 或 'ar'
    'use_gcc': False,
    'is_undirected': True,
    'device': 'cuda'
}

env = build_environment(env_cfg)

# 使用合成图
env_cfg = {
    'type': 'NetworkDismantlingEnv',
    'synth_type': 'ba',           # 'ba', 'er', 'ws'
    'synth_args': {
        'n': 100,
        'm': 2
    },
    'node_features': 'combin',
    'value_type': 'auc',
    'device': 'cuda'
}

env = build_environment(env_cfg)
```

#### 环境参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `graph` | nx.Graph | None | 网络图对象 |
| `synth_type` | str | 'ba' | 合成图类型 |
| `node_features` | str | 'ones' | 节点特征类型 |
| `value_type` | str | 'auc' | 奖励类型 ('auc', 'ar') |
| `use_gcc` | bool | False | 只与最大连通分支交互 |
| `use_component` | bool | False | 是否使用连通分量 |
| `is_undirected` | bool | True | 是否无向图 |

#### 使用示例

```python
# 重置环境
state = env.reset()

# 状态包含：
# - edge_index: 边索引张量
# - node_features: 节点特征
# - node_mask: 节点掩码
# - num_nodes: 节点数
# - reward_info: 奖励信息

# 选择一个节点进行移除
action = 5  # 要移除的节点索引
next_state, reward, done, info = env.step(action)

# info 包含：
# - 'lcc_size': 当前最大连通分量大小
# - 'attack_rate': 攻击率
# - 'remaining_nodes': 剩余节点数

# 获取最大连通分量大小
lcc_size = env.lcc()

# 获取最大连通分量的节点列表
lcc_nodes = env.lcc_component()
```

### VectorizedEnv

向量化环境，支持并行运行多个环境实例。

#### 初始化

```python
from src.utils import build_environment

# 方式1：创建多个副本
env_cfg = {
    'type': 'NetworkDismantlingEnv',
    'synth_type': 'ba',
    'synth_args': {'n': 100, 'm': 2},
    'env_num': 4  # 创建4个副本
}

venv = build_environment(env_cfg)

# 方式2：从图列表创建
env_cfg = {
    'type': 'NetworkDismantlingEnv',
    'graph_list': [graph1, graph2, graph3],
    'common_kwargs': {
        'node_features': 'combin',
        'value_type': 'auc'
    }
}

venv = build_environment(env_cfg)

# 方式3：从配置列表创建
env_cfg = {
    'type': 'NetworkDismantlingEnv',
    'env_kwargs_list': [
        {'graph': graph1},
        {'graph': graph2},
        {'graph': graph3}
    ]
}

venv = build_environment(env_cfg)
```

#### 使用示例

```python
# 重置所有环境
states = venv.reset()

# 批量执行动作
actions = [5, 10, 15, 20]  # 每个环境的动作
next_states, rewards, dones, infos = venv.step(actions)

# 访问单个环境
single_env = venv[0]

# 获取环境数量
num_envs = len(venv)
```

---

## 模型模块 (models)

### 概述

模型模块包含图神经网络的主干网络、预测头和完整模型。

### 主干网络 (backbones)

#### GraphSAGE

```python
from centrilearn.utils import build_backbone

backbone_cfg = {
    'type': 'GraphSAGE',
    'in_channels': 2,
    'hidden_channels': 64,
    'num_layers': 3,
    'output_dim': None,          # 可选，输出维度
    'aggr': 'mean',              # 聚合方式: 'mean', 'max', 'sum'
    'graph_aggr': 'add',         # 图池化: 'add', 'mean', 'max'
    'norm': None,                # 归一化: 'batch', 'layer', None
    'dropout': 0.0
}

backbone = build_backbone(backbone_cfg)

# 前向传播
# 输入: data = {'node_features': ..., 'edge_index': ...}
# 输出: {'node_embed': ..., 'graph_embed': ...}
output = backbone(data)
```

#### GAT (Graph Attention Network)

```python
backbone_cfg = {
    'type': 'GAT',
    'in_channels': 2,
    'hidden_channels': 64,
    'num_layers': 3,
    'heads': 4,                  # 注意力头数
    'concat': True,              # 是否拼接多头
    'v2': False,                 # 使用 GATv2
    # ... 其他参数同 GraphSAGE
}

backbone = build_backbone(backbone_cfg)
```

#### GIN (Graph Isomorphism Network)

```python
backbone_cfg = {
    'type': 'GIN',
    'in_channels': 2,
    'hidden_channels': 64,
    'num_layers': 3,
    # ... 其他参数同 GraphSAGE
}

backbone = build_backbone(backbone_cfg)
```

#### DeepNet (ResNet 风格)

```python
backbone_cfg = {
    'type': 'DeepNet',
    'in_channels': 2,
    'hidden_channels': 64,
    'num_blocks': 3,             # Block 数量
    'block_config': {
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.1
    },
    'use_residual': True,        # 使用残差连接
    'nn': 'GraphSAGE'            # 基础 GNN 类型
}

backbone = build_backbone(backbone_cfg)
```

#### FPNet (Feature Pyramid Network)

```python
backbone_cfg = {
    'type': 'FPNet',
    'in_channels': 2,
    'hidden_channels_list': [64, 128, 256],  # 各层隐藏维度
    'num_layers_list': [2, 2, 2],            # 各层 GNN 层数
    'fusion_mode': 'add',                    # 'add', 'concat', 'attention'
    'nn': 'GraphSAGE'
}

backbone = build_backbone(backbone_cfg)
```

### 预测头 (heads)

#### QHead (Q 值头)

```python
from centrilearn.utils import build_head

q_head_cfg = {
    'type': 'QHead',
    'in_channels': 64,
    'hidden_layers': [128, 64],  # 可选
    'activation': 'leaky_relu',
    'dropout': 0.0
}

q_head = build_head(q_head_cfg)

# 输入: node_embed [batch_size, hidden_channels]
# 输出: q_values [batch_size, 1]
```

#### PolicyHead (策略头)

```python
policy_head_cfg = {
    'type': 'PolicyHead',
    'in_channels': 64,
    'hidden_layers': [128, 64]
}

policy_head = build_head(policy_head_cfg)

# 输出: logits [batch_size, 1]
```

#### VHead (价值头)

```python
v_head_cfg = {
    'type': 'VHead',
    'in_channels': 64
}

v_head = build_head(v_head_cfg)

# 输出: v_values [batch_size, 1]
```

#### MLPHead (通用 MLP)

```python
mlp_head_cfg = {
    'type': 'MLPHead',
    'in_channels': 64,
    'hidden_layers': [128, 64, 32],
    'activation': 'leaky_relu',
    'dropout': 0.1,
    'norm': 'layer'
}

mlp_head = build_head(mlp_head_cfg)
```

### 完整模型 (network_dismantler)

#### Qnet

```python
from centrilearn.utils import build_network_dismantler

model_cfg = {
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
}

model = build_network_dismantler(model_cfg)

# 输入: data = {'node_features': ..., 'edge_index': ...}
# 输出: {'q_values': ..., 'node_embed': ..., 'graph_embed': ...}
output = model(data)
```

#### ActorCritic

```python
model_cfg = {
    'type': 'ActorCritic',
    'backbone_cfg': {
        'type': 'GraphSAGE',
        'in_channels': 2,
        'hidden_channels': 64,
        'num_layers': 3
    },
    'actor_head_cfg': {
        'type': 'PolicyHead',
        'in_channels': 64
    },
    'critic_head_cfg': {
        'type': 'VHead',
        'in_channels': 64
    },
    'num_critics': 1             # Critic 数量
}

model = build_network_dismantler(model_cfg)

# 输出: {'logit': ..., 'v_values': ...}
```

---

## 缓冲区模块 (buffer)

### 概述

缓冲区模块用于存储和管理训练经验。

### ReplayBuffer

DQN 使用的经验回放缓冲区。

#### 初始化

```python
from centrilearn.utils import build_replaybuffer

buffer_cfg = {
    'type': 'ReplayBuffer',
    'capacity': 10000,
    'n_step': 5,                 # N 步回报
    'gamma': 0.99,
    'prioritized': False          # 是否使用优先级采样
}

# 优先级采样
buffer_cfg = {
    'type': 'ReplayBuffer',
    'capacity': 10000,
    'n_step': 5,
    'prioritized': True,
    'alpha': 0.6,                # 优先度指数
    'beta_start': 0.4,           # 重要性采样初始 beta
    'beta_frames': 100000        # beta 衰减帧数
}

buffer = build_replaybuffer(buffer_cfg)
```

#### 使用示例

```python
# 添加经验
buffer.push(state, action, reward, next_state, done)

# 采样一批数据
batch, indices, weights = buffer.sample(batch_size=32)

# batch 包含:
# - states, actions, rewards, next_states, dones
# - weights: 重要性采样权重（优先级采样时）

# 更新优先级（优先级采样时）
priorities = compute_priorities(...)
buffer.update_priorities(indices, priorities)

# 获取当前 beta
beta = buffer.get_beta()

# 清空缓冲区
buffer.clear()
```

### RolloutBuffer

PPO 使用的轨迹缓冲区。

#### 初始化

```python
buffer_cfg = {
    'type': 'RolloutBuffer',
    'capacity': 2048
}

buffer = build_replaybuffer(buffer_cfg)
```

#### 使用示例

```python
# 添加经验
buffer.push(
    state=state,
    action=action,
    log_prob=log_prob,
    reward=reward,
    done=done,
    value=value
)

# 获取训练批次（计算 GAE 优势）
batches = buffer.get_batches(
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95
)

# 每个 batch 包含:
# - states, actions, log_probs, rewards, dones, values
# - advantages, returns

# 清空缓冲区
buffer.clear()
```

---

## 指标模块 (metrics)

### 概述

指标模块用于评估训练过程中的性能。

### BaseMetric 基类

所有指标的基类，定义了统一的接口。

#### 使用示例

```python
from centrilearn.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__(name='Custom', record='max')

    def process(self, state, action, reward, next_state, done, info=None):
        # 处理单个步骤
        return some_value

    def evaluate(self, env=None, model=None, num_episodes=1):
        # 在完整 episode 上评估
        return {'result': 0.5}

    def compute(self):
        # 计算当前累积值
        return self._total / self._count
```

### AUC (Area Under Curve)

计算 Attack Curve 的面积。

#### 初始化

```python
from centrilearn.utils import build_metric

metric_cfg = {
    'type': 'AUC',
    'name': 'AUC',
    'record': 'min'              # 记录最小值
}

metric = build_metric(metric_cfg)
```

### AttackRate

计算攻击率（行动次数/节点数）。

#### 初始化

```python
metric_cfg = {
    'type': 'AttackRate',
    'name': 'AttackRate',
    'record': 'min'
}

metric = build_metric(metric_cfg)
```

### MetricManager

指标管理器，用于管理多个指标。

#### 初始化

```python
from centrilearn.utils import build_metric_manager

manager_cfg = {
    'metrics': [
        {'type': 'AUC', 'record': 'min'},
        {'type': 'AttackRate', 'record': 'min'}
    ],
    'save_dir': './logs/metrics',
    'log_interval': 10
}

manager = build_metric_manager(manager_cfg)
```

#### 使用示例

```python
# 更新所有指标
results = manager.update(state, action, reward, next_state, done, info)

# 评估所有指标
results = manager.evaluate(env, model, num_episodes=5)

# 获取所有指标结果
results = manager.get_results()

# 获取摘要（仅当前值）
summary = manager.get_summary()

# 保存结果
manager.save('path/to/metrics.json')

# 加载结果
manager.load('path/to/metrics.json')

# 打印日志
manager.log(step=100, prefix="Train")

# 添加新指标
from src.metrics import BaseMetric
manager.add_metric(CustomMetric())

# 重置所有指标
manager.reset()
```

---

## 工具模块 (utils)

### 概述

工具模块提供了注册器机制、构建函数和训练入口。

### Registry 注册器

#### 使用示例

```python
from centrilearn.utils.registry import BACKBONES

# 注册自定义模块
@BACKBONES.register_module()
class MyBackbone(nn.Module):
    def __init__(self, ...):
        pass

# 使用注册的模块
from src.utils import build_backbone
backbone = build_backbone({'type': 'MyBackbone', ...})
```

### 构建函数

#### build_optimizer

```python
from centrilearn.utils import build_optimizer

optimizer_cfg = {
    'type': 'Adam',
    'lr': 0.0001,
    'weight_decay': 0.0005
}

optimizer = build_optimizer(model, optimizer_cfg)

# 支持的优化器: Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta
```

#### build_scheduler

```python
from centrilearn.utils import build_scheduler

scheduler_cfg = {
    'type': 'CosineAnnealingLR',
    'T_max': 1000,
    'eta_min': 1e-6
}

scheduler = build_scheduler(optimizer, scheduler_cfg)

# 支持的调度器:
# - StepLR, MultiStepLR, ExponentialLR
# - CosineAnnealingLR, CosineAnnealingWarmRestarts
# - ReduceLROnPlateau, LinearLR, CyclicLR
# - OneCycleLR, LambdaLR, etc.
```

#### build_from_cfg

```python
from centrilearn.utils import build_from_cfg
from src.utils.registry import BACKBONES

cfg = {
    'type': 'GraphSAGE',
    'in_channels': 2,
    'hidden_channels': 64,
    'num_layers': 3
}

backbone = build_from_cfg(cfg, BACKBONES)
```

### 训练入口

#### train_from_cfg

```python
import yaml
from centrilearn.utils import train_from_cfg

# 加载配置文件
with open('configs/network_dismantling/dqn.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 开始训练
results, algorithm = train_from_cfg(config, verbose=True)

# 访问训练结果
print(f"Average reward: {results['avg_reward']:.4f}")
print(f"Total episodes: {results['total_episodes']}")
```

---

## 最佳实践

### 1. 模型选择

- **简单任务**: 使用 GraphSAGE + 简单的 Head
- **需要注意力**: 使用 GAT
- **图同构性**: 使用 GIN

### 2. 算法选择

- **离散动作空间**: 使用 DQN
- **连续/离散动作空间**: 使用 PPO
- **需要样本效率**: 使用优先级 ReplayBuffer
- **稳定性**: 使用 PPO

### 3. 环境配置

- **快速迭代**: 使用小规模合成图 (n=30-50)
- **最终评估**: 使用真实图或大规模图 (n=100+)
- **并行训练**: 使用 VectorizedEnv (env_num=4-8)

### 4. 训练技巧

- 使用检查点保存和恢复
- 记录多个指标进行综合评估
- 定期评估避免过拟合
- 使用学习率调度器优化收敛

### 5. 调试建议

- 从小规模开始，验证代码正确性
- 打印中间结果，检查数据流
- 使用 TensorBoard 可视化训练过程
- 单元测试确保各模块正常工作

---

## 常见问题

### Q: 如何添加自定义算法？

A: 继承 BaseAlgorithm，实现抽象方法，然后使用 @ALGORITHMS.register_module() 注册。

### Q: 如何使用自己的图数据？

A: 使用 networkx 加载图数据，然后通过 build_environment 创建环境。

### Q: 如何调试训练过程？

A: 使用 metric_manager 记录指标，设置 verbose=True 打印日志，或者使用 TensorBoard。

### Q: 如何加速训练？

A: 使用向量化环境、GPU 训练、增大 batch_size、减小模型复杂度。

---

更多详细信息，请参考 [API 文档](api_reference.md) 和 [示例代码](../examples/)。
