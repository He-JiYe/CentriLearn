"""
强化学习算法基类
定义算法的标准接口和通用功能
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from src.utils import build_optimizer, build_scheduler


class BaseAlgorithm(ABC):
    """强化学习算法基类

    定义算法的通用接口，包括训练、评估、保存/加载等功能。

    Attributes:
        model: 模型实例或模型参数
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        device: 运行设备
        training_step: 当前训练步数
        model_cfg: 模型配置
    """

    def __init__(self,
                 model: Union[nn.Module, Dict[str, Any]],
                 optimizer_cfg: Dict[str, Any],
                 scheduler_cfg: Optional[Dict[str, Any]] = None,
                 device: str = 'cpu'):
        """初始化算法

        Args:
            model: 神经网络模型实例 或 模型配置字典
                   如果是字典，将自动构建模型
            optimizer_cfg: 优化器配置，例如 {'type': 'Adam', 'lr': 1e-4}
            scheduler_cfg: 学习率调度器配置，例如 {'type': 'StepLR', 'step_size': 100}
            device: 运行设备
        """
        self.device = device

        if isinstance(model, nn.Module):
            self.model = model.to(device)
            self.model_cfg = None
        elif isinstance(model, dict):
            self.model_cfg = model
            self.model = self._build_model(model).to(device)
        else:
            raise TypeError(f"model 必须是 nn.Module 或 Dict，当前类型: {type(model)}")

        # 构建优化器和调度器
        self.optimizer = build_optimizer(self.model, optimizer_cfg)
        self.scheduler = build_scheduler(self.optimizer, scheduler_cfg)

        # 训练状态
        self.training_step = 0

    @abstractmethod
    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """从配置构建模型

        子类应该重写此方法以定义特定的模型构建逻辑。

        Args:
            model_cfg: 模型配置字典

        Returns:
            构建好的模型实例
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一步训练
        
        Args:
            batch: 训练数据批次
            
        Returns:
            训练指标字典
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """更新模型参数

        Returns:
            更新指标字典
        """
        pass

    @abstractmethod
    def select_action(self, state: Dict[str, Any], **kwargs) -> Any:
        """选择动作

        Args:
            state: 当前状态
            **kwargs: 算法特定的参数（如 epsilon, deterministic）

        Returns:
            选择动作的相关信息（动作本身以及可能的额外信息）
        """
        pass

    @abstractmethod
    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """收集经验到缓冲区

        Args:
            state: 当前状态
            *args: 其他必需参数（如 action, reward, next_state, done, log_prob, value 等）
            **kwargs: 可选参数
        """
        pass
    
    def set_train_mode(self):
        """设置为训练模式"""
        self.model.train()
    
    def set_eval_mode(self):
        """设置为评估模式"""
        self.model.eval()
    
    def save_checkpoint(self, path: str, **kwargs):
        """保存检查点
        
        Args:
            path: 保存路径
            **kwargs: 额外保存的信息
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            **kwargs
        }
        
        # 保存调度器状态（如果存在）
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            检查点字典
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        
        # 恢复调度器状态（如果存在）
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def step_scheduler(self, metrics: Optional[Dict[str, float]] = None):
        """更新学习率调度器
        
        Args:
            metrics: 指标字典（某些调度器如 ReduceLROnPlateau 需要）
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    metric_value = next(iter(metrics.values()))
                    self.scheduler.step(metric_value)
            else:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率
        
        Returns:
            当前学习率
        """
        return self.optimizer.param_groups[0]['lr']
    
    def get_model(self):
        """获取模型"""
        return self.model
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, device={self.device})"
