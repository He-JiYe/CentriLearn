"""
训练入口函数
支持通过配置文件自动构建环境、算法、模型并进行训练
"""
import src 
from typing import Dict, Any, Tuple

def train_from_cfg(config: Dict[str, Any],
                   verbose: bool = True,
                   **kwargs) -> Tuple[Dict[str, Any], Any]:
    """从配置文件进行训练

    这是整个框架的统一训练入口，类似于 mmcv 的配置化训练风格。
    支持通过配置文件自动构建环境、算法、模型、优化器、调度器等组件。
    训练循环由 train.py 中的函数统一实现，不依赖算法实例的方法。

    Args:
        config: 训练配置字典，包含以下键：
            - algorithm: 算法配置
            - environment: 环境配置
            - training: 训练参数

        verbose: 是否打印训练日志
        **kwargs: 额外的训练参数，覆盖 config['training'] 中的配置

    Returns:
        results: 训练结果字典，包含训练指标等信息
        algorithm: 训练完成的算法实例
    """
    # 检查参数
    if not isinstance(config, dict):
        raise TypeError(f'config must be a dict, but got {type(config)}')

    required_keys = ['algorithm', 'environment', 'training']
    for key in required_keys:
        if key not in config:
            raise KeyError(f'config must contain "{key}", but got keys: {config.keys()}')

    algorithm_required_keys = ['type', 'model', 'optimizer_cfg', 'replaybuffer_cfg', 'algo_cfg', 'device']
    for key in algorithm_required_keys:
        if key not in config['algorithm']:
            raise KeyError(f'algorithm config must contain "{key}", but got keys: {config.keys()}')

    algorithm_cfg, env_cfg, training_cfg = config['algorithm'], config['environment'], config['training']

    if verbose:
        print("\n" + "=" * 70)
        print(f"开始训练 - 算法类型: {algorithm_cfg['type']}")
        print("=" * 70)

    # 1. 构建环境
    if verbose:
        print(f"\n[1/4] 构建环境: {env_cfg.get('type', 'unknown')}")

    # 延迟导入构建函数
    from src.utils.builder import build_environment
    env = build_environment(env_cfg)

    if verbose:
        print(f"      [OK] 环境构建完成: {env}")

    # 3. 构建算法
    if verbose:
        print(f"\n[3/4] 构建算法: {algorithm_cfg['type']}")
        print(f"      - 模型类型: {algorithm_cfg.get('model', 'unknown')}")
        print(f"      - 优化器: {algorithm_cfg.get('optimizer_cfg', {}).get('type')} (lr={algorithm_cfg.get('optimizer_cfg', {}).get('lr', 'N/A')})")

    # 延迟导入避免循环依赖
    from src.utils.registry import ALGORITHMS
    from src.utils.builder import build_algorithm
    # 检查算法是否已注册
    if algorithm_cfg['type'] not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm type: {algorithm_cfg['type']}. "
                        f"Available algorithms: {list(ALGORITHMS.module_dict.keys())}")

    algorithm = build_algorithm(algorithm_cfg)

    if verbose:
        print(f"      [OK] 算法构建完成: {algorithm}")

    # 4. 执行训练
    final_training_cfg = {**training_cfg, **kwargs}

    if verbose:
        print(f"\n[4/4] 开始训练...")
        print(f"      训练配置: {final_training_cfg}")

    results = algorithm._run_training_loop(env, final_training_cfg)

    # 5. 训练完成
    if verbose:
        print(f"\n[5/4] 训练完成！")
        print(f"\n训练结果:")
        for key, value in results.items():
            if key != 'episode_rewards':  # 避免打印过长的列表
                print(f"  {key}: {value}")
        print("=" * 70 + "\n")

    return results, algorithm


if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open('configs/network_dismantling/dqn.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    # 开始训练
    results, algo = train_from_cfg(config, verbose=True)