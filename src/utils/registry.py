"""
通用注册器
支持模块的注册和动态构建
"""
import inspect
from typing import Type, Dict


class Registry:
    """注册器类

    将字符串映射到类，支持通过配置动态构建对象。

    Args:
        name (str): 注册器名称
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
        """获取注册的类

        Args:
            key (str): 类名字符串

        Returns:
            Type: 对应的类
        """
        if key not in self._module_dict:
            raise KeyError(f'{key} is not in the {self._name} registry')
        return self._module_dict[key]

    def _register_module(self, module_class: Type, module_name: str = None,
                        force: bool = False) -> None:
        """注册模块

        Args:
            module_class: 要注册的模块类
            module_name: 注册的模块名称，如果未指定则使用类名
            force: 是否覆盖已存在的同名类
        """
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self._name}')

        self._module_dict[module_name] = module_class

    def register_module(self, name: str = None, force: bool = False, module: Type = None):
        """注册模块

        在 `self._module_dict` 中添加记录，key 是类名或指定的名称，value 是类本身。

        可以用作装饰器或普通函数。

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


# 创建模型注册器
NN = Registry('nn')
BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NETWORK_DISMANTLER = Registry('network_dismantler')

# 创建环境注册器
ENVIRONMENTS = Registry('environments')

# 创建算法注册器
ALGORITHMS = Registry('algorithms')
