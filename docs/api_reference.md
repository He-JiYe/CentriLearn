# CentriLearn API Reference

本文档提供 CentriLearn 所有公共 API 的详细参考。

## 目录

- [算法 API (algorithms)](#算法-api-algorithms)
- [环境 API (environments)](#环境-api-environments)
- [模型 API (models)](#模型-api-models)
- [缓冲区 API (buffer)](#缓冲区-api-buffer)
- [指标 API (metrics)](#指标-api-metrics)
- [工具 API (utils)](#工具-api-utils)

---

## 算法 API (algorithms)

### BaseAlgorithm

所有强化学习算法的基类。

```python
class BaseAlgorithm
```

#### Methods

##### `__init__(model, optimizer_cfg, scheduler_cfg=None, replaybuffer_cfg=None, metric_manager_cfg=None, device='cpu')`

初始化算法。

**Parameters:**
- `model` (nn.Module): 模型实例
- `optimizer_cfg` (dict): 优化器配置
- `scheduler_cfg` (dict, optional): 学习率调度器配置
- `replaybuffer_cfg` (dict, optional): 经验缓冲区配置
- `metric_manager_cfg` (dict, optional): 指标管理器配置
- `device` (str): 运行设备 ('cpu' 或 'cuda')

**Raises:**
- `TypeError`: 如果 optimizer_cfg 不是字典

---

##### `set_train_mode()`

将模型设置为训练模式。

**Returns:** None

---

##### `set_eval_mode()`

将模型设置为评估模式。

**Returns:** None

---

##### `save_checkpoint(path, **kwargs)`

保存训练检查点。

**Parameters:**
- `path` (str): 保存路径
- `**kwargs`: 额外保存的信息（如 episode, best_reward 等）

**Returns:** None

---

##### `load_checkpoint(path)`

加载训练检查点。

**Parameters:**
- `path` (str): 检查点文件路径

**Returns:**
- `dict`: 检查点数据

**Raises:**
- `FileNotFoundError`: 如果文件不存在

---

##### `step_scheduler(metrics=None)`

更新学习率调度器。

**Parameters:**
- `metrics` (dict, optional): 评估指标（用于 ReduceLROnPlateau）

**Returns:** None

---

##### `get_lr()`

获取当前学习率。

**Returns:**
- `float`: 当前学习率

---

##### `get_model()`

获取模型。

**Returns:**
- `nn.Module`: 模型实例

---

##### `_build_model(model_cfg) -> nn.Module`

从配置构建模型（抽象方法，子类必须实现）。

**Parameters:**
- `model_cfg` (dict): 模型配置

**Returns:**
- `nn.Module`: 模型实例

---

##### `train_step(batch) -> dict`

执行一步训练（抽象方法，子类必须实现）。

**Parameters:**
- `batch` (dict): 训练批次数据

**Returns:**
- `dict`: 训练损失信息

---

##### `update(*args, **kwargs) -> dict`

更新模型参数（抽象方法，子类必须实现）。

**Returns:**
- `dict`: 更新信息

---

##### `select_action(state, **kwargs) -> any`

选择动作（抽象方法，子类必须实现）。

**Parameters:**
- `state` (dict): 当前状态
- `**kwargs`: 额外参数

**Returns:**
- `any`: 选择的动作

---

##### `collect_experience(state, *args, **kwargs)`

收集经验到缓冲区（抽象方法，子类必须实现）。

**Parameters:**
- `state` (dict): 当前状态
- `*args`, `**kwargs`: 经验数据

**Returns:** None

---

### DQN

Deep Q-Network 算法。

```python
class DQN(BaseAlgorithm)
```

#### Methods

##### `__init__(model_cfg, optimizer_cfg, scheduler_cfg=None, replaybuffer_cfg=None, metric_manager_cfg=None, algo_cfg=None, device='cpu')`

初始化 DQN 算法。

**Parameters:**
- `model_cfg` (dict): 模型配置
- `optimizer_cfg` (dict): 优化器配置
- `scheduler_cfg` (dict, optional): 学习率调度器配置
- `replaybuffer_cfg` (dict, optional): 经验缓冲区配置
- `metric_manager_cfg` (dict, optional): 指标管理器配置
- `algo_cfg` (dict, optional): DQN 特定配置
  - `gamma` (float): 折扣因子，默认 0.99
  - `epsilon_start` (float): 初始探索率，默认 1.0
  - `epsilon_end` (float): 最终探索率，默认 0.01
  - `epsilon_decay` (int): 探索率衰减步数，默认 10000
  - `tau` (float): 软更新系数，默认 0.005
  - `rcst_coef` (float): 重建损失系数，默认 0.0001
- `device` (str): 运行设备

---

##### `compute_epsilon() -> float`

计算当前探索率。

**Returns:**
- `float`: 当前 epsilon 值

---

##### `select_action(states, epsilon=None) -> tuple`

使用 epsilon-greedy 策略选择动作。

**Parameters:**
- `states` (dict): 当前状态
- `epsilon` (float, optional): 指定 epsilon，None 则自动计算

**Returns:**
- `tuple`: (actions, epsilon)

---

##### `update(batch_size) -> dict`

从缓冲区采样并更新模型。

**Parameters:**
- `batch_size` (int): 批次大小

**Returns:**
- `dict`: 包含以下键:
  - `q_loss`: Q 损失
  - `reconstruction_loss`: 重建损失
  - `total_loss`: 总损失

---

##### `get_q_values(state) -> torch.Tensor`

获取 Q 值。

**Parameters:**
- `state` (dict): 当前状态

**Returns:**
- `torch.Tensor`: Q 值张量

---

##### `collect_experience(states, actions, rewards, next_states, dones)`

收集经验到缓冲区。

**Parameters:**
- `states` (dict): 状态
- `actions` (torch.Tensor): 动作
- `rewards` (torch.Tensor): 奖励
- `next_states` (dict): 下一状态
- `dones` (torch.Tensor): 终止标志

**Returns:** None

---

### PPO

Proximal Policy Optimization 算法。

```python
class PPO(BaseAlgorithm)
```

#### Methods

##### `__init__(model_cfg, optimizer_cfg, scheduler_cfg=None, replaybuffer_cfg=None, metric_manager_cfg=None, algo_cfg=None, device='cpu')`

初始化 PPO 算法。

**Parameters:**
- `model_cfg` (dict): 模型配置
- `optimizer_cfg` (dict): 优化器配置
- `scheduler_cfg` (dict, optional): 学习率调度器配置
- `replaybuffer_cfg` (dict, optional): 经验缓冲区配置
- `metric_manager_cfg` (dict, optional): 指标管理器配置
- `algo_cfg` (dict, optional): PPO 特定配置
  - `gamma` (float): 折扣因子，默认 0.99
  - `gae_lambda` (float): GAE lambda 参数，默认 0.95
  - `clip_epsilon` (float): PPO 裁剪参数，默认 0.2
  - `entropy_coef` (float): 熵正则化系数，默认 0.01
  - `value_coef` (float): 价值损失系数，默认 0.5
  - `max_grad_norm` (float): 最大梯度裁剪，默认 0.5
  - `num_epochs` (int): 每次更新的 epoch 数，默认 1
  - `rcst_coef` (float): 重建损失系数，默认 0.0001
- `device` (str): 运行设备

---

##### `select_action(state, deterministic=False) -> tuple`

选择动作。

**Parameters:**
- `state` (dict): 当前状态
- `deterministic` (bool): 是否确定性选择

**Returns:**
- `tuple`: (action, log_prob, value)

---

##### `update(batch_size=64) -> dict`

更新模型。

**Parameters:**
- `batch_size` (int): 批次大小

**Returns:**
- `dict`: 包含以下键:
  - `policy_loss`: 策略损失
  - `value_loss`: 价值损失
  - `entropy_loss`: 熵损失
  - `total_loss`: 总损失

---

##### `get_action_value(state) -> tuple`

获取动作和价值（推理模式）。

**Parameters:**
- `state` (dict): 当前状态

**Returns:**
- `tuple`: (action, value)

---

##### `collect_experience(state, action, log_prob, reward, done, value)`

收集经验到缓冲区。

**Parameters:**
- `state` (dict): 状态
- `action` (torch.Tensor): 动作
- `log_prob` (torch.Tensor): 对数概率
- `reward` (torch.Tensor): 奖励
- `done` (torch.Tensor): 终止标志
- `value` (torch.Tensor): 价值

**Returns:** None

---

## 环境 API (environments)

### BaseEnv

所有环境的基类。

```python
class BaseEnv
```

#### Methods

##### `__init__(graph=None, synth_type='ba', synth_args=None, node_features='ones', use_component=False, is_undirected=True, device='cpu')`

初始化环境。

**Parameters:**
- `graph` (nx.Graph, optional): 网络图对象
- `synth_type` (str): 合成图类型 ('ba', 'er', 'ws')
- `synth_args` (dict, optional): 合成图参数
- `node_features` (str): 节点特征类型 ('ones', 'degree', 'combin')
- `use_component` (bool): 是否使用连通分量
- `is_undirected` (bool): 是否无向图
- `device` (str): 计算设备

---

##### `reset(graph=None) -> dict`

重置环境。

**Parameters:**
- `graph` (nx.Graph, optional): 新的图对象

**Returns:**
- `dict`: 初始状态，包含:
  - `edge_index`: 边索引张量
  - `node_features`: 节点特征
  - `node_mask`: 节点掩码
  - `num_nodes`: 节点数
  - `reward_info`: 奖励信息

---

##### `get_state() -> dict`

获取当前状态。

**Returns:**
- `dict`: 当前状态

---

##### `get_pyg_data(mask=None) -> torch_geometric.data.Data`

获取 PyG 格式数据。

**Parameters:**
- `mask` (torch.Tensor, optional): 节点掩码

**Returns:**
- `torch_geometric.data.Data`: PyG 数据对象

---

##### `connected_components(edge_index, num_nodes) -> list`

计算连通分量。

**Parameters:**
- `edge_index` (torch.Tensor): 边索引
- `num_nodes` (int): 节点数

**Returns:**
- `list`: 连通分量列表

---

##### `is_empty() -> bool`

检查图是否为空。

**Returns:**
- `bool`: 是否为空

---

##### `step(action, mapping) -> tuple`

执行一步动作（抽象方法）。

**Parameters:**
- `action` (int): 动作
- `mapping` (dict): 节点映射

**Returns:**
- `tuple`: (next_state, reward, done, info)

---

### NetworkDismantlingEnv

网络瓦解环境。

```python
class NetworkDismantlingEnv(BaseEnv)
```

#### Methods

##### `__init__(graph=None, synth_type='ba', synth_args=None, node_features='ones', value_type='auc', use_gcc=False, use_component=False, is_undirected=True, device='cpu')`

初始化网络瓦解环境。

**Parameters:**
- `graph` (nx.Graph, optional): 网络图对象
- `synth_type` (str): 合成图类型
- `synth_args` (dict, optional): 合成图参数
- `node_features` (str): 节点特征类型
- `value_type` (str): 奖励类型 ('auc', 'ar')
- `use_gcc` (bool): 只与最大连通分支交互
- `use_component` (bool): 是否使用连通分量
- `is_undirected` (bool): 是否无向图
- `device` (str): 计算设备

---

##### `step(action, mapping) -> tuple`

执行一步动作。

**Parameters:**
- `action` (int): 要移除的节点索引
- `mapping` (dict): 节点映射

**Returns:**
- `tuple`: (next_state, reward, done, info)
  - `next_state`: 下一状态
  - `reward`: 奖励
  - `done`: 终止标志
  - `info`: 额外信息
    - `lcc_size`: 当前最大连通分量大小
    - `attack_rate`: 攻击率
    - `remaining_nodes`: 剩余节点数

---

##### `lcc() -> int`

返回剩余图的最大连通分量大小。

**Returns:**
- `int`: 最大连通分量大小

---

##### `lcc_component() -> list`

返回最大连通分量的节点索引。

**Returns:**
- `list`: 节点索引列表

---

##### `remove_node(node, mapping)`

移除节点。

**Parameters:**
- `node` (int): 节点索引
- `mapping` (dict): 节点映射

**Returns:** None

---

### VectorizedEnv

向量化环境。

```python
class VectorizedEnv
```

#### Methods

##### `__init__(env_class, env_kwargs, env_num=None)`

初始化向量化环境。

**Parameters:**
- `env_class`: 环境类
- `env_kwargs` (list): 环境参数列表
- `env_num` (int, optional): 环境数量

---

##### `from_single_config(env_class, env_kwargs, env_num) -> VectorizedEnv`

从单个配置创建多个副本。

**Parameters:**
- `env_class`: 环境类
- `env_kwargs` (dict): 环境参数
- `env_num` (int): 环境数量

**Returns:**
- `VectorizedEnv`: 向量化环境实例

---

##### `from_graph_list(env_class, graph_list, common_kwargs=None) -> VectorizedEnv`

从图列表创建向量化环境。

**Parameters:**
- `env_class`: 环境类
- `graph_list` (list): 图列表
- `common_kwargs` (dict, optional): 通用参数

**Returns:**
- `VectorizedEnv`: 向量化环境实例

---

##### `reset(indices=None) -> list`

重置环境。

**Parameters:**
- `indices` (list, optional): 要重置的环境索引

**Returns:**
- `list`: 观测列表

---

##### `step(actions) -> tuple`

批量执行动作。

**Parameters:**
- `actions` (list): 动作列表

**Returns:**
- `tuple`: (observations, rewards, dones, infos)

---

##### `__len__() -> int`

返回环境数量。

**Returns:**
- `int`: 环境数量

---

##### `__getitem__(index) -> BaseEnv`

获取单个环境。

**Parameters:**
- `index` (int): 环境索引

**Returns:**
- `BaseEnv`: 环境实例

---

## 模型 API (models)

### GraphSAGE

GraphSAGE 主干网络。

```python
class GraphSAGE
```

#### Methods

##### `__init__(in_channels, hidden_channels, num_layers, output_dim=None, aggr='mean', graph_aggr='add', norm=None, dropout=0.0)`

初始化 GraphSAGE。

**Parameters:**
- `in_channels` (int): 输入特征维度
- `hidden_channels` (int): 隐藏特征维度
- `num_layers` (int): GNN 层数
- `output_dim` (int, optional): 输出维度
- `aggr` (str): 聚合方式 ('mean', 'max', 'sum')
- `graph_aggr` (str): 图池化方式 ('add', 'mean', 'max')
- `norm` (str, optional): 归一化方式 ('batch', 'layer')
- `dropout` (float): Dropout 概率

---

##### `forward(data) -> dict`

前向传播。

**Parameters:**
- `data` (dict): 输入数据，包含:
  - `node_features`: 节点特征
  - `edge_index`: 边索引

**Returns:**
- `dict`: 输出，包含:
  - `node_embed`: 节点嵌入
  - `graph_embed`: 图嵌入

---

### GAT

Graph Attention Network。

```python
class GAT
```

#### Methods

##### `__init__(in_channels, hidden_channels, num_layers, output_dim=None, aggr='mean', graph_aggr='add', norm=None, dropout=0.0, v2=False, heads=1, concat=True)`

初始化 GAT。

**Parameters:**
- `in_channels` (int): 输入特征维度
- `hidden_channels` (int): 隐藏特征维度
- `num_layers` (int): GNN 层数
- `output_dim` (int, optional): 输出维度
- `aggr` (str): 聚合方式
- `graph_aggr` (str): 图池化方式
- `norm` (str, optional): 归一化方式
- `dropout` (float): Dropout 概率
- `v2` (bool): 使用 GATv2
- `heads` (int): 注意力头数
- `concat` (bool): 是否拼接多头

---

### GIN

Graph Isomorphism Network。

```python
class GIN
```

#### Methods

##### `__init__(in_channels, hidden_channels, num_layers, output_dim=None, aggr='mean', graph_aggr='add', norm=None, dropout=0.0)`

初始化 GIN。

**Parameters:** 与 GraphSAGE 相同

---

### DeepNet

ResNet 风格深度网络。

```python
class DeepNet
```

#### Methods

##### `__init__(in_channels, hidden_channels=64, num_blocks=3, block_config=None, aggr='mean', graph_aggr='add', norm='layer', dropout=0.0, use_residual=True, output_dim=None, nn='GraphSAGE')`

初始化 DeepNet。

**Parameters:**
- `in_channels` (int): 输入特征维度
- `hidden_channels` (int): 隐藏特征维度
- `num_blocks` (int): Block 数量
- `block_config` (dict, optional): Block 配置
- `aggr` (str): GraphSAGE 聚合方式
- `graph_aggr` (str): 图池化方式
- `norm` (str): 归一化类型
- `dropout` (float): Dropout 概率
- `use_residual` (bool): 是否使用残差连接
- `output_dim` (int, optional): 输出维度
- `nn` (str): 基础 GNN 类型

---

### FPNet

Feature Pyramid Network。

```python
class FPNet
```

#### Methods

##### `__init__(in_channels, hidden_channels_list=[64, 128, 256], num_layers_list=None, aggr='mean', graph_aggr='add', norm='layer', dropout=0.0, fusion_mode='add', output_dim=None, nn='GraphSAGE')`

初始化 FPNet。

**Parameters:**
- `in_channels` (int): 输入特征维度
- `hidden_channels_list` (list): 各层隐藏维度列表
- `num_layers_list` (list, optional): 各层 GNN 层数列表
- `aggr` (str): GraphSAGE 聚合方式
- `graph_aggr` (str): 图池化方式
- `norm` (str): 归一化类型
- `dropout` (float): Dropout 概率
- `fusion_mode` (str): 特征融合方式 ('add', 'concat', 'attention')
- `output_dim` (int, optional): 输出维度
- `nn` (str): 基础 GNN 类型

---

### QHead

Q 值预测头。

```python
class QHead
```

#### Methods

##### `__init__(in_channels, hidden_layers=None, activation='leaky_relu', dropout=0.0)`

初始化 QHead。

**Parameters:**
- `in_channels` (int): 输入特征维度
- `hidden_layers` (list, optional): 隐藏层维度列表
- `activation` (str): 激活函数
- `dropout` (float): Dropout 概率

---

##### `forward(node_embed) -> torch.Tensor`

前向传播。

**Parameters:**
- `node_embed` (torch.Tensor): 节点嵌入

**Returns:**
- `torch.Tensor`: Q 值

---

### PolicyHead

策略预测头。

```python
class PolicyHead
```

#### Methods

##### `__init__(in_channels, hidden_layers=None, activation='leaky_relu', dropout=0.0)`

初始化 PolicyHead。

**Parameters:** 与 QHead 相同

---

##### `forward(node_embed) -> torch.Tensor`

前向传播。

**Parameters:**
- `node_embed` (torch.Tensor): 节点嵌入

**Returns:**
- `torch.Tensor`: 动作 logits

---

### VHead

价值预测头。

```python
class VHead
```

#### Methods

##### `__init__(in_channels, hidden_layers=None, activation='leaky_relu', dropout=0.0)`

初始化 VHead。

**Parameters:** 与 QHead 相同

---

##### `forward(node_embed) -> torch.Tensor`

前向传播。

**Parameters:**
- `node_embed` (torch.Tensor): 节点嵌入

**Returns:**
- `torch.Tensor`: 价值估计

---

### Qnet

Q 网络。

```python
class Qnet
```

#### Methods

##### `__init__(backbone_cfg, q_head_cfg=None)`

初始化 Qnet。

**Parameters:**
- `backbone_cfg` (dict): 主干网络配置
- `q_head_cfg` (dict, optional): Q 值头配置

---

##### `forward(data) -> dict`

前向传播。

**Parameters:**
- `data` (dict): 输入数据

**Returns:**
- `dict`: 包含以下键:
  - `q_values`: Q 值
  - `node_embed`: 节点嵌入
  - `graph_embed`: 图嵌入

---

### ActorCritic

Actor-Critic 网络。

```python
class ActorCritic
```

#### Methods

##### `__init__(backbone_cfg, actor_head_cfg=None, critic_head_cfg=None, num_critics=1)`

初始化 ActorCritic。

**Parameters:**
- `backbone_cfg` (dict): 主干网络配置
- `actor_head_cfg` (dict, optional): Actor 头配置
- `critic_head_cfg` (dict, optional): Critic 头配置
- `num_critics` (int): Critic 数量

---

##### `forward(data) -> dict`

前向传播。

**Parameters:**
- `data` (dict): 输入数据

**Returns:**
- `dict`: 包含以下键:
  - `logit`: 动作 logits
  - `v_values`: 价值估计

---

## 缓冲区 API (buffer)

### ReplayBuffer

经验回放缓冲区。

```python
class ReplayBuffer
```

#### Methods

##### `__init__(capacity, n_step=1, gamma=0.99, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-6, prioritized=False)`

初始化 ReplayBuffer。

**Parameters:**
- `capacity` (int): 缓冲区容量
- `n_step` (int): N 步回报步数
- `gamma` (float): 折扣因子
- `alpha` (float): 优先度指数
- `beta_start` (float): 重要性采样初始 beta
- `beta_frames` (int): beta 衰减帧数
- `epsilon` (float): 最小优先度
- `prioritized` (bool): 是否使用优先级采样

---

##### `push(state, action, reward, next_state, done)`

添加经验。

**Parameters:**
- `state` (dict): 状态
- `action` (torch.Tensor): 动作
- `reward` (torch.Tensor): 奖励
- `next_state` (dict): 下一状态
- `done` (torch.Tensor): 终止标志

**Returns:** None

---

##### `sample(batch_size) -> tuple`

采样一批数据。

**Parameters:**
- `batch_size` (int): 批次大小

**Returns:**
- `tuple`: (batch, indices, weights)
  - `batch`: 批次数据
  - `indices`: 采样索引
  - `weights`: 重要性采样权重

---

##### `update_priorities(indices, priorities)`

更新优先级。

**Parameters:**
- `indices` (list): 索引列表
- `priorities` (list): 优先度列表

**Returns:** None

---

##### `get_beta() -> float`

获取当前 beta 值。

**Returns:**
- `float`: 当前 beta

---

##### `__len__() -> int`

获取缓冲区大小。

**Returns:**
- `int`: 当前经验数量

---

##### `clear()`

清空缓冲区。

**Returns:** None

---

### RolloutBuffer

轨迹缓冲区。

```python
class RolloutBuffer
```

#### Methods

##### `__init__(capacity)`

初始化 RolloutBuffer。

**Parameters:**
- `capacity` (int): 缓冲区容量

---

##### `push(state, action, log_prob, reward, done, value)`

添加经验。

**Parameters:**
- `state` (dict): 状态
- `action` (torch.Tensor): 动作
- `log_prob` (torch.Tensor): 对数概率
- `reward` (torch.Tensor): 奖励
- `done` (torch.Tensor): 终止标志
- `value` (torch.Tensor): 价值

**Returns:** None

---

##### `get_batches(batch_size, gamma=0.99, gae_lambda=0.95) -> list`

获取训练批次（计算 GAE 优势）。

**Parameters:**
- `batch_size` (int): 批次大小
- `gamma` (float): 折扣因子
- `gae_lambda` (float): GAE lambda 参数

**Returns:**
- `list`: 批次列表，每个批次包含:
  - `states`, `actions`, `log_probs`, `rewards`, `dones`, `values`
  - `advantages`, `returns`

---

##### `clear()`

清空缓冲区。

**Returns:** None

---

##### `__len__() -> int`

获取缓冲区大小。

**Returns:**
- `int`: 当前经验数量

---

## 指标 API (metrics)

### BaseMetric

所有指标的基类。

```python
class BaseMetric
```

#### Methods

##### `__init__(name=None, record='max')`

初始化指标。

**Parameters:**
- `name` (str, optional): 指标名称
- `record` (str): 记录方式 ('max' 或 'min')

---

##### `update(value)`

更新指标累积值。

**Parameters:**
- `value` (float): 指标值

**Returns:** None

---

##### `reset()`

重置指标状态。

**Returns:** None

---

##### `get_result() -> dict`

获取指标结果。

**Returns:**
- `dict`: 包含以下键:
  - `value`: 当前值
  - `max`: 最大值
  - `min`: 最小值
  - `count`: 计数
  - `history`: 历史值

---

##### `process(state, action, reward, next_state, done, info=None) -> float`

处理单个步骤（抽象方法）。

**Parameters:**
- `state` (dict): 状态
- `action`: 动作
- `reward` (float): 奖励
- `next_state` (dict): 下一状态
- `done` (bool): 终止标志
- `info` (dict, optional): 额外信息

**Returns:**
- `float`: 指标值

---

##### `evaluate(env=None, model=None, num_episodes=1) -> dict`

在完整 episode 上评估（抽象方法）。

**Parameters:**
- `env`: 环境
- `model`: 模型
- `num_episodes` (int): Episode 数量

**Returns:**
- `dict`: 评估结果

---

##### `compute() -> float`

计算当前累积值（抽象方法）。

**Returns:**
- `float`: 当前值

---

### AUC

Attack Curve 面积指标。

```python
class AUC(BaseMetric)
```

#### Methods

##### `__init__(name='AUC', record='min')`

初始化 AUC 指标。

---

##### `process(state, action, reward, next_state, done, info=None) -> float`

处理单个步骤。

**Returns:**
- `float`: 累积奖励

---

### AttackRate

攻击率指标。

```python
class AttackRate(BaseMetric)
```

#### Methods

##### `__init__(name='AttackRate', record='min')`

初始化 AttackRate 指标。

---

### MetricManager

指标管理器。

```python
class MetricManager
```

#### Methods

##### `__init__(metrics=None, save_dir=None, log_interval=100)`

初始化指标管理器。

**Parameters:**
- `metrics` (list, optional): 指标列表
- `save_dir` (str, optional): 保存目录
- `log_interval` (int): 日志打印间隔

---

##### `add_metric(metric)`

添加单个指标。

**Parameters:**
- `metric` (BaseMetric): 指标实例

**Returns:** None

---

##### `add_metrics(metrics)`

添加多个指标。

**Parameters:**
- `metrics` (list): 指标列表

**Returns:** None

---

##### `remove_metric(name) -> bool`

移除指标。

**Parameters:**
- `name` (str): 指标名称

**Returns:**
- `bool`: 是否成功移除

---

##### `get_metric(name) -> BaseMetric`

获取指标实例。

**Parameters:**
- `name` (str): 指标名称

**Returns:**
- `BaseMetric`: 指标实例

---

##### `update(state, action, reward, next_state, done, info=None) -> dict`

更新所有指标。

**Parameters:**
- `state` (dict): 状态
- `action`: 动作
- `reward` (float): 奖励
- `next_state` (dict): 下一状态
- `done` (bool): 终止标志
- `info` (dict, optional): 额外信息

**Returns:**
- `dict`: 更新结果

---

##### `evaluate(env=None, model=None, num_episodes=1) -> dict`

评估所有指标。

**Parameters:**
- `env`: 环境
- `model`: 模型
- `num_episodes` (int): Episode 数量

**Returns:**
- `dict`: 评估结果

---

##### `get_results() -> dict`

获取所有指标结果。

**Returns:**
- `dict`: 指标结果字典

---

##### `get_summary() -> dict`

获取摘要（仅当前值）。

**Returns:**
- `dict`: 摘要字典

---

##### `reset()`

重置所有指标。

**Returns:** None

---

##### `reset_metric(name)`

重置指定指标。

**Parameters:**
- `name` (str): 指标名称

**Returns:** None

---

##### `save(path=None)`

保存指标结果。

**Parameters:**
- `path` (str, optional): 保存路径

**Returns:** None

---

##### `load(path)`

加载指标结果。

**Parameters:**
- `path` (str): 文件路径

**Returns:** None

---

##### `log(step=None, prefix="")`

打印指标日志。

**Parameters:**
- `step` (int, optional): 当前步数
- `prefix` (str): 日志前缀

**Returns:** None

---

##### `__len__() -> int`

返回指标数量。

**Returns:**
- `int`: 指标数量

---

## 工具 API (utils)

### Registry

注册器类。

```python
class Registry
```

#### Methods

##### `__init__(name)`

初始化注册器。

**Parameters:**
- `name` (str): 注册器名称

---

##### `get(key) -> type`

获取注册的类。

**Parameters:**
- `key` (str): 类名字符串

**Returns:**
- `type`: 对应的类

**Raises:**
- `KeyError`: 如果 key 不存在

---

##### `register_module(name=None, force=False, module=None)`

注册模块（装饰器或函数）。

**Parameters:**
- `name` (str, optional): 注册名称
- `force` (bool): 是否覆盖已存在的类
- `module` (type, optional): 要注册的类

**Returns:**
- 装饰器函数或注册后的类

---

##### `__len__() -> int`

返回注册数量。

**Returns:**
- `int`: 注册数量

---

### 构建函数

#### build_optimizer(model, cfg) -> torch.optim.Optimizer

构建优化器。

**Parameters:**
- `model` (nn.Module): 模型
- `cfg` (dict): 优化器配置
  - `type`: 优化器类型 ('Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta')
  - `lr`: 学习率
  - `weight_decay`: 权重衰减
  - 其他优化器特定参数

**Returns:**
- `torch.optim.Optimizer`: 优化器实例

---

#### build_scheduler(optimizer, cfg=None) -> torch.optim.lr_scheduler._LRScheduler

构建学习率调度器。

**Parameters:**
- `optimizer` (torch.optim.Optimizer): 优化器
- `cfg` (dict, optional): 调度器配置
  - `type`: 调度器类型
  - 调度器特定参数

**Returns:**
- `torch.optim.lr_scheduler._LRScheduler`: 调度器实例

**支持的调度器类型:**
- `StepLR`, `MultiStepLR`, `ExponentialLR`
- `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`
- `ReduceLROnPlateau`, `LinearLR`, `CyclicLR`
- `OneCycleLR`, `LambdaLR`, `MultiplicativeLR`
- `ConstantLR`, `SequentialLR`, `ChainedScheduler`

---

#### build_from_cfg(cfg, registry, default_args=None)

从配置构建模块。

**Parameters:**
- `cfg` (dict): 配置字典
  - `type`: 类名字符串或类
  - 其他初始化参数
- `registry` (Registry): 注册器
- `default_args` (dict, optional): 默认参数

**Returns:**
- 构建的对象实例

---

#### build_backbone(cfg, default_args=None)

构建主干网络。

**Parameters:**
- `cfg` (dict or list): 主干网络配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 主干网络实例

---

#### build_head(cfg, default_args=None)

构建预测头。

**Parameters:**
- `cfg` (dict or list): 预测头配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 预测头实例

---

#### build_network_dismantler(cfg, default_args=None)

构建网络瓦解模型。

**Parameters:**
- `cfg` (dict or list): 模型配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 模型实例

---

#### build_environment(cfg, default_args=None)

构建环境。

**Parameters:**
- `cfg` (dict): 环境配置
  - `type`: 环境类型
  - 其他环境参数
  - `env_num` (optional): 向量化环境数量
  - `graph_list` (optional): 图列表
  - `env_kwargs_list` (optional): 配置列表
- `default_args` (dict, optional): 默认参数

**Returns:**
- 环境实例（单环境或向量化环境）

---

#### build_algorithm(cfg, default_args=None)

构建算法。

**Parameters:**
- `cfg` (dict or list): 算法配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 算法实例

---

#### build_replaybuffer(cfg, default_args=None)

构建经验缓冲区。

**Parameters:**
- `cfg` (dict or list): 缓冲区配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 缓冲区实例

---

#### build_metric(cfg, default_args=None)

构建指标。

**Parameters:**
- `cfg` (dict or list): 指标配置
- `default_args` (dict, optional): 默认参数

**Returns:**
- 指标实例或实例列表

---

#### build_metric_manager(cfg=None)

构建指标管理器。

**Parameters:**
- `cfg` (dict, optional): 指标管理器配置
  - `metrics`: 指标配置列表
  - `save_dir`: 保存目录
  - `log_interval`: 日志间隔

**Returns:**
- `MetricManager`: 指标管理器实例

---

### 训练函数

#### train_from_cfg(config, verbose=True, **kwargs) -> tuple

从配置训练。

**Parameters:**
- `config` (dict): 配置字典
  - `algorithm`: 算法配置
  - `environment`: 环境配置
  - `training`: 训练参数
- `verbose` (bool): 是否打印日志
- `**kwargs`: 额外参数

**Returns:**
- `tuple`: (results, algorithm)
  - `results` (dict): 训练结果
  - `algorithm`: 训练完成的算法实例

---

## 预定义注册器

以下注册器在 `src.utils.registry` 中预定义：

```python
NN                     # 基础图神经网络层
BACKBONES              # 主干网络
HEADS                  # 预测头
NETWORK_DISMANTLER     # 网络瓦解模型
ENVIRONMENTS           # 环境
ALGORITHMS             # 算法
REPLAYBUFFERS          # 经验缓冲区
METRICS                # 指标
```

## 注册模块列表

### NN
- `GAT`
- `GIN`
- `GraphSAGE`

### BACKBONES
- `GAT`
- `GIN`
- `GraphSAGE`
- `DeepNet`
- `FPNet`

### HEADS
- `MLPHead`
- `QHead`
- `PolicyHead`
- `VHead`
- `DuelingHead`
- `ComponentValueHead`

### NETWORK_DISMANTLER
- `Qnet`
- `ActorCritic`

### ENVIRONMENTS
- `NetworkDismantlingEnv`

### ALGORITHMS
- `DQN`
- `PPO`

### REPLAYBUFFERS
- `ReplayBuffer`
- `RolloutBuffer`

### METRICS
- `AUC`
- `AttackRate`
