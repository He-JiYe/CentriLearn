"""
Registry for model.
"""
import inspect
from typing import Type, Dict


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
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
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            Type: The corresponding class.
        """
        if key not in self._module_dict:
            raise KeyError(f'{key} is not in the {self._name} registry')
        return self._module_dict[key]

    def _register_module(self, module_class: Type, module_name: str = None,
                          force: bool = False) -> None:
        """Register a module.

        Args:
            module_class: Module class to be registered.
            module_name: Module name to be registered. If not specified, the class name will be used.
            force: Whether to override an existing class with the same name.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self._name}')

        self._module_dict[module_name] = module_class

    def register_module(self, name: str = None, force: bool = False, module: Type = None):
        """Register a module.

        A record will be added to `self._module_dict`, the key is the class name
        or the specified name, and the value is the class itself.

        It can be used as a decorator or a normal function.

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
BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NETWORK_DISMANTLER = Registry('network_dismantler')