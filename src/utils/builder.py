"""
优化器和调度器构建器
支持通过配置文件动态创建优化器和调度器
"""
import torch
from torch import nn
import inspect
from typing import Dict, Any, Optional, Union, List, Tuple
from .registry import BACKBONES, HEADS, NETWORK_DISMANTLER, ENVIRONMENTS, ALGORITHMS
from .buffer import ReplayBuffer, RolloutBuffer


def build_replaybuffer(cfg: Dict[str, Any]) -> Union[ReplayBuffer, RolloutBuffer]:
    """从配置构建缓冲区

    Args:
        cfg: 缓冲区配置字典
            - type: 缓冲区类型 ('replay' 或 'rollout')
            - capacity: 缓冲区容量
            - n_step: N-step 回报步数（仅 replay）
            - gamma: 折扣因子（仅 replay，默认从算法获取）
            - alpha: 优先度指数（仅 replay，默认 0.6）
            - beta_start: 初始 beta（仅 replay，默认 0.4）
            - beta_frames: beta 线性增加帧数（仅 replay，默认 100000）
            - prioritized: 是否使用优先度采样（仅 replay，默认 False）

    Returns:
        构建的缓冲区实例

    Example:
        >>> # ReplayBuffer 配置
        >>> cfg = {
        ...     'type': 'replay',
        ...     'capacity': 10000,
        ...     'n_step': 3,
        ...     'prioritized': True
        ... }
        >>> buffer = build_replaybuffer(cfg)

        >>> # RolloutBuffer 配置
        >>> cfg = {
        ...     'type': 'rollout',
        ...     'capacity': 2048
        ... }
        >>> buffer = build_replaybuffer(cfg)
    """
    buffer_type = cfg.get('type', 'replay')

    if buffer_type == 'replay':
        # 构建 ReplayBuffer
        return ReplayBuffer(
            capacity=cfg.get('capacity', 10000),
            n_step=cfg.get('n_step', 1),
            gamma=cfg.get('gamma', 0.99),
            alpha=cfg.get('alpha', 0.6),
            beta_start=cfg.get('beta_start', 0.4),
            beta_frames=cfg.get('beta_frames', 100000),
            prioritized=cfg.get('prioritized', False)
        )

    elif buffer_type == 'rollout':
        # 构建 RolloutBuffer
        return RolloutBuffer(
            capacity=cfg.get('capacity', 2048)
        )

    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}. Must be 'replay' or 'rollout'")


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """构建优化器
    
    Args:
        model: 神经网络模型
        cfg: 优化器配置字典
        
    Returns:
        优化器实例
        
    Example:
        >>> optimizer_cfg = {
        ...     'type': 'Adam',
        ...     'lr': 1e-4,
        ...     'weight_decay': 1e-5
        ... }
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    optimizer_type = cfg.get('type', 'Adam')
    lr = cfg.get('lr', 1e-4)
    weight_decay = cfg.get('weight_decay', 0)

    # 移除 type, lr, weight_decay 键
    params = {k: v for k, v in cfg.items() if k not in ['type', 'lr', 'weight_decay']}
    
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **params)
    
    elif optimizer_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **params)
    
    elif optimizer_type == 'SGD':
        momentum = params.get('momentum', 0.9)
        nesterov = params.get('nesterov', False)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            **{k: v for k, v in params.items() if k not in ['momentum', 'nesterov']}
        )
    
    elif optimizer_type == 'RMSprop':
        alpha = params.get('alpha', 0.99)
        momentum = params.get('momentum', 0)
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            weight_decay=weight_decay,
            **{k: v for k, v in params.items() if k not in ['alpha', 'momentum']}
        )
    
    elif optimizer_type == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **params
        )
    
    elif optimizer_type == 'Adadelta':
        rho = params.get('rho', 0.9)
        return torch.optim.Adadelta(
            model.parameters(),
            lr=lr,
            rho=rho,
            weight_decay=weight_decay,
            **{k: v for k, v in params.items() if k != 'rho'}
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, 
                    cfg: Optional[Dict[str, Any]] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """构建学习率调度器
    
    Args:
        optimizer: 优化器实例
        cfg: 调度器配置字典，如果为 None 则不创建调度器
        
    Returns:
        学习率调度器实例，如果 cfg 为 None 则返回 None
        
    Example:
        >>> # 线性衰减
        >>> scheduler_cfg = {
        ...     'type': 'LinearLR',
        ...     'total_iters': 1000
        ... }
        >>> scheduler = build_scheduler(optimizer, scheduler_cfg)
        
        >>> # 余弦退火
        >>> scheduler_cfg = {
        ...     'type': 'CosineAnnealingLR',
        ...     'T_max': 1000,
        ...     'eta_min': 1e-6
        ... }
        >>> scheduler = build_scheduler(optimizer, scheduler_cfg)
        
        >>> # 自定义衰减
        >>> scheduler_cfg = {
        ...     'type': 'LambdaLR',
        ...     'lambda_fn': lambda epoch: 0.99 ** epoch
        ... }
        >>> scheduler = build_scheduler(optimizer, scheduler_cfg)
    """
    if cfg is None:
        return None
    
    scheduler_type = cfg.get('type')
    
    # 提取通用参数
    params = {k: v for k, v in cfg.items() if k != 'type'}
    
    if scheduler_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 10),
            gamma=params.get('gamma', 0.1),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.get('milestones', [50, 100]),
            gamma=params.get('gamma', 0.1),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get('gamma', 0.95),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', 100),
            eta_min=params.get('eta_min', 0),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get('T_0', 10),
            T_mult=params.get('T_mult', 1),
            eta_min=params.get('eta_min', 0),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.1),
            patience=params.get('patience', 10),
            threshold=params.get('threshold', 1e-4),
            threshold_mode=params.get('threshold_mode', 'rel'),
            cooldown=params.get('cooldown', 0),
            min_lr=params.get('min_lr', 0),
            eps=params.get('eps', 1e-8),
            verbose=params.get('verbose', False)
        )
    
    elif scheduler_type == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=params.get('start_factor', 1.0),
            end_factor=params.get('end_factor', 0.0),
            total_iters=params.get('total_iters', 100),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'CyclicLR':
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=params.get('base_lr', 1e-6),
            max_lr=params.get('max_lr', 1e-3),
            step_size_up=params.get('step_size_up', 2000),
            step_size_down=params.get('step_size_down', None),
            mode=params.get('mode', 'triangular'),
            gamma=params.get('gamma', 1.0),
            scale_fn=params.get('scale_fn', None),
            scale_mode=params.get('scale_mode', 'exp'),
            cycle_momentum=params.get('cycle_momentum', True),
            base_momentum=params.get('base_momentum', 0.8),
            max_momentum=params.get('max_momentum', 0.9),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.get('max_lr', 1e-3),
            total_steps=params.get('total_steps', None),
            epochs=params.get('epochs', 100),
            steps_per_epoch=params.get('steps_per_epoch', None),
            pct_start=params.get('pct_start', 0.3),
            anneal_strategy=params.get('anneal_strategy', 'cos'),
            div_factor=params.get('div_factor', 25),
            final_div_factor=params.get('final_div_factor', 1e4),
            three_phase=params.get('three_phase', False),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'LambdaLR':
        # 支持通过配置传递 lambda 函数
        lambda_fn = params.get('lambda_fn')
        if lambda_fn is None:
            raise ValueError("LambdaLR requires 'lambda_fn' parameter")
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda_fn,
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'MultiplicativeLR':
        lr_lambda = params.get('lr_lambda')
        if lr_lambda is None:
            raise ValueError("MultiplicativeLR requires 'lr_lambda' parameter")
        return torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'ConstantLR':
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=params.get('factor', 1.0),
            total_iters=params.get('total_iters', 100),
            last_epoch=params.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'SequentialLR':
        schedulers = params.get('schedulers', [])
        milestones = params.get('milestones', [])
        last_epoch = params.get('last_epoch', -1)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch
        )
    
    elif scheduler_type == 'ChainedScheduler':
        schedulers = params.get('schedulers', [])
        return torch.optim.lr_scheduler.ChainedScheduler(schedulers)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def build_from_cfg(cfg: Dict, registry, default_args: Dict = None):
    """从配置字典构建模块

    Args:
        cfg: 配置字典，必须包含 'type' 键
        registry: 用于搜索类型的注册器
        default_args: 默认初始化参数

    Returns:
        obj: 构建的对象

    Example:
        >>> cfg = {'type': 'SimpleNet', 'input_dim': 10, 'hidden_dim': 64}
        >>> backbone = build_from_cfg(cfg, BACKBONES)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')

    args = cfg.copy()

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)


def build_backbone(cfg: Union[Dict, List], default_args: Dict = None):
    """从配置构建 backbone

    Args:
        cfg: backbone 配置，可以是字典或字典列表
        default_args: 默认参数

    Returns:
        构建的 backbone 模块
    """
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, BACKBONES, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, BACKBONES, default_args)


def build_head(cfg: Union[Dict, List], default_args: Dict = None):
    """从配置构建 head

    Args:
        cfg: head 配置，可以是字典或字典列表
        default_args: 默认参数

    Returns:
        构建的 head 模块
    """
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, HEADS, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, HEADS, default_args)


def build_network_dismantler(cfg: Union[Dict, List], default_args: Dict = None):
    """从配置构建 network_dismantler

    Args:
        cfg: network_dismantler 配置，可以是字典或字典列表
        default_args: 默认参数

    Returns:
        构建的 network_dismantler 模块
    """
    if isinstance(cfg, list):
        return nn.Sequential(*[
            build_from_cfg(_cfg, NETWORK_DISMANTLER, default_args) for _cfg in cfg
        ])
    return build_from_cfg(cfg, NETWORK_DISMANTLER, default_args)


def build_environment(cfg: Dict, default_args: Dict = None):
    """从配置构建环境

    Args:
        cfg: 环境配置，可以是字典或字典列表
        default_args: 默认参数

    Returns:
        构建的环境实例
    """
    env_class = cfg.get('type', 'NetworkDismantlingEnv')
    env_class = ENVIRONMENTS.get(env_class) if isinstance(env_class, str) else env_class
    
    # 检查是否为向量化环境配置
    if cfg.get('graph_list', []):
        from ..environments import VectorizedEnv
        graph_list = cfg.get('graph_list', [])
        common_kwargs = cfg.get('common_kwargs', {})

        return VectorizedEnv.from_graph_list(env_class, graph_list, common_kwargs)
    
    if cfg.get('env_kwargs_list', []):
        from ..environments import VectorizedEnv
        env_kwargs_list = cfg.get('env_kwargs_list', [])
        return VectorizedEnv(env_class, env_kwargs_list)

    return build_from_cfg(cfg, ENVIRONMENTS, default_args)

def build_algorithm(cfg: Union[Dict, List], default_args: Dict = None):
    """从配置构建算法

    Args:
        cfg: 算法配置，可以是字典或字典列表
        default_args: 默认参数

    Returns:
        构建的算法实例
    """
    if isinstance(cfg, list):
        return [build_from_cfg(_cfg, ALGORITHMS, default_args) for _cfg in cfg]
    return build_from_cfg(cfg, ALGORITHMS, default_args)

