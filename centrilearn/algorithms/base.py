"""
Reinforcement Learning Algorithm Base Class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from centrilearn.utils import (
    build_metric_manager,
    build_optimizer,
    build_replaybuffer,
    build_scheduler,
)


class BaseAlgorithm(ABC):
    """Reinforcement Learning Algorithm Base Class

    Define the common interface of an algorithm, including training, evaluation, saving/loading, etc.

    Attributes:
        model: module instance or model config dict
        optimizer: optimizer config dict
        scheduler: learning rate scheduler config dict (optional)
        replaybuffer: replay buffer config dict
        metric_manager: metric manager config dict (optional)
        device: device to run the algorithm
        training_step: training step
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        replaybuffer_cfg: Optional[Dict[str, Any]] = None,
        metric_manager_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """Initialize the algorithm.

        Args:
            model_cfg: Model configuration dictionary
            optimizer_cfg: Optimizer configuration, e.g., {'type': 'Adam', 'lr': 1e-4}
            scheduler_cfg: Learning rate scheduler configuration (optional)
            replaybuffer_cfg: Experience replay buffer configuration
            metric_manager_cfg: Metric manager configuration (optional)
            device: Device to run on
        """
        self.device = device
        self.model_cfg = model_cfg
        self.model = self._build_model(model_cfg).to(device)
        self.optimizer = (
            build_optimizer(self.model, optimizer_cfg) if optimizer_cfg else None
        )
        self.scheduler = (
            build_scheduler(self.optimizer, scheduler_cfg) if scheduler_cfg else None
        )
        self.replay_buffer = (
            build_replaybuffer(replaybuffer_cfg) if replaybuffer_cfg else None
        )
        self.metric_manager = (
            build_metric_manager(metric_manager_cfg) if metric_manager_cfg else None
        )

        # Training state
        self.training_step = 0

    @abstractmethod
    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """Build model from configuration.

        Subclasses should add corresponding models in the models directory
        based on specific tasks, use registries and builders to extend,
        and finally call the builder from here to construct models.

        Args:
            model_cfg: Model configuration dictionary

        Returns:
            Built model instance
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Update model parameters.

        Subclasses should implement parameter update logic based on reinforcement learning algorithms.

        Returns:
            Update metrics dictionary
        """
        pass

    @abstractmethod
    def select_action(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[Union[torch.Tensor, int], ...]:
        """Select action.

        Subclasses should implement action selection logic based on reinforcement learning algorithms.

        Args:
            state: Current state
            **kwargs: Algorithm-specific parameters (e.g., epsilon, deterministic)

        Returns:
            Action-related information (action itself and possibly additional information)
        """
        pass

    @abstractmethod
    def collect_experience(self, state: Dict[str, Any], *args, **kwargs):
        """Collect experience to buffer.

        Args:
            state: Current state
            *args: Other required args (e.g., action, reward, next_state, done, log_prob, value)
            **kwargs: Optional args
        """
        pass

    def set_train_mode(self) -> None:
        """Set to training mode."""
        self.model.train()

    def set_eval_mode(self) -> None:
        """Set to evaluation mode."""
        self.model.eval()

    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint.

        Args:
            path: Save path
            **kwargs: Additional information to save
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            **kwargs,
        }

        # Save scheduler state if exists
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint.

        Args:
            path: Checkpoint path

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        # Restore optimizer and scheduler state if exists
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def step_scheduler(self, metrics: Optional[Dict[str, float]] = None):
        """Update learning rate scheduler.

        Args:
            metrics: Metrics dictionary (some schedulers like ReduceLROnPlateau require this)
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    metric_value = next(iter(metrics.values()))
                    self.scheduler.step(metric_value)
            else:
                self.scheduler.step()

    @abstractmethod
    def learn(
        self,
        env: Any,
        training_cfg: Dict[str, Any],
        verbose: bool = True,
        logger=None,
        tensorboard_writer=None,
    ) -> Dict[str, Any]:
        """Generic training loop implementation.

        Args:
            env: Environment instance
            training_cfg: Training configuration
            verbose: Whether to print logs
            logger: Logger object
            tensorboard_writer: TensorBoard writer object

        Returns:
            Training results dictionary
        """
        pass

    @abstractmethod
    def rollout(self, env: Any) -> Dict[str, Any]:
        """Generic test/rollout interface.

        Args:
            env: Environment instance

        Returns:
            Test results dictionary
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, device={self.device})"
