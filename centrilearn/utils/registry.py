"""
Universal Registry
Supports module registration and dynamic construction
"""

import inspect
from typing import Dict, Type


class Registry:
    """Registry Class

    Maps strings to classes, supports dynamic object construction via configuration.

    Args:
        name (str): Registry name
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

    def get(self, key: str) -> Type:
        """Get registered class.

        Args:
            key (str): Class name string

        Returns:
            Type: Corresponding class
        """
        if key not in self._module_dict:
            raise KeyError(f"{key} not found in {self._name}")
        return self._module_dict[key]

    def _register_module(
        self, module_class: Type, module_name: str = None, force: bool = False
    ) -> None:
        """Register module.

        Args:
            module_class: Module class to register
            module_name: Module name for registration, uses class name if not specified
            force: Whether to overwrite existing class with same name
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self._name}")

        self._module_dict[module_name] = module_class

    def register_module(
        self, name: str = None, force: bool = False, module: Type = None
    ):
        """Register module.

        Add record to `self._module_dict`, key is class name or specified name, value is the class itself.

        Can be used as a decorator or regular function.

        Example:
            >>> backbones = Registry('backbone')
            >>>
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>>
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>>
            >>> @backbones.register_module(force=True)
            >>> class ResNet:
            >>>     pass
        """

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        if module is not None:
            return _register(module)

        return _register


# Create model registry
NN = Registry("nn")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
NETWORK_DISMANTLER = Registry("network_dismantler")

# Create environment registry
ENVIRONMENTS = Registry("environments")

# Create algorithm registry
ALGORITHMS = Registry("algorithms")

# Create replay buffer registry
REPLAYBUFFERS = Registry("replaybuffers")

# Create metric registry
METRICS = Registry("metrics")
