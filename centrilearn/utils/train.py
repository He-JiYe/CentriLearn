"""
训练入口函数
支持通过配置文件自动构建环境、算法、模型并进行训练
"""

from typing import Any, Dict, Tuple

import centrilearn


def train_from_cfg(
    config: Dict[str, Any], verbose: bool = True, **kwargs
) -> Tuple[Dict[str, Any], Any]:
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
        raise TypeError(f"config must be a dict, but got {type(config)}")

    required_keys = ["algorithm", "environment", "training"]
    for key in required_keys:
        if key not in config:
            raise KeyError(
                f'config must contain "{key}", but got keys: {config.keys()}'
            )

    algorithm_required_keys = [
        "type",
        "model",
        "optimizer_cfg",
        "replaybuffer_cfg",
        "algo_cfg",
        "device",
    ]
    for key in algorithm_required_keys:
        if key not in config["algorithm"]:
            raise KeyError(
                f'algorithm config must contain "{key}", but got keys: {config["algorithm"].keys()}'
            )

    algorithm_cfg, env_cfg, training_cfg = (
        config["algorithm"],
        config["environment"],
        config["training"],
    )

    if verbose:
        print("\n" + "=" * 70)
        print(f"开始训练 - 算法类型: {algorithm_cfg['type']}")
        print("=" * 70)

    # 1. 构建环境
    if verbose:
        print(f"\n[1/5] 构建环境: {env_cfg.get('type', 'unknown')}")

    # 延迟导入构建函数
    from centrilearn.utils.builder import build_environment

    env = build_environment(env_cfg)

    # 确定环境数量
    env_num = 1
    try:
        # 检查是否为向量化环境
        from centrilearn.environments import VectorizedEnv

        if isinstance(env, VectorizedEnv):
            env_num = env.num_envs
    except ImportError:
        pass

    if verbose:
        print(f"      [OK] 环境构建完成: {env}")
        print(f"      环境数量: {env_num}")

    # 3. 构建算法
    if verbose:
        print(f"\n[2/5] 构建算法: {algorithm_cfg['type']}")
        print(f"      - 模型类型: {algorithm_cfg.get('model', 'unknown')}")
        print(
            f"      - 优化器: {algorithm_cfg.get('optimizer_cfg', {}).get('type')} (lr={algorithm_cfg.get('optimizer_cfg', {}).get('lr', 'N/A')})"
        )

    # 延迟导入避免循环依赖
    from centrilearn.utils.builder import build_algorithm
    from centrilearn.utils.registry import ALGORITHMS

    # 检查算法是否已注册
    if algorithm_cfg["type"] not in ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm type: {algorithm_cfg['type']}. "
            f"Available algorithms: {list(ALGORITHMS.module_dict.keys())}"
        )

    # 将环境数量传递给算法配置
    if "replaybuffer_cfg" in algorithm_cfg:
        # 为 replaybuffer 添加 env_num 参数
        if isinstance(algorithm_cfg["replaybuffer_cfg"], dict):
            algorithm_cfg["replaybuffer_cfg"]["env_num"] = env_num
    if "metric_manager_cfg" in algorithm_cfg:
        if isinstance(algorithm_cfg["metric_manager_cfg"], dict):
            algorithm_cfg["metric_manager_cfg"]["num_env"] = env_num

    algorithm = build_algorithm(algorithm_cfg)

    if verbose:
        print(f"      [OK] 算法构建完成: {algorithm}")

    # 4. 检查是否需要恢复训练
    resume_from = training_cfg.get("resume")
    if resume_from:
        if verbose:
            print(f"\n[3/5] 恢复训练...")
            print(f"      从检查点恢复: {resume_from}")

        # 加载检查点
        try:
            checkpoint = algorithm.load_checkpoint(resume_from)
            if verbose:
                print(f"      [OK] 检查点加载成功")
                print(f"      训练步数: {algorithm.training_step}")
                if "episode" in checkpoint:
                    print(f"      恢复episode: {checkpoint['episode']}")
        except Exception as e:
            if verbose:
                print(f"      [警告] 检查点加载失败: {e}")
                print(f"      将从头开始训练")

    # 5. 执行训练
    final_training_cfg = {**training_cfg, **kwargs}

    if verbose:
        if not resume_from:
            print(f"\n[4/5] 开始训练...")
        else:
            print(f"\n[4/5] 继续训练...")
        print(f"      训练配置: {final_training_cfg}")

    results = algorithm._run_training_loop(env, final_training_cfg)

    # 6. 训练完成
    if verbose:
        print(f"\n[5/5] 训练完成！")
        print(f"\n训练结果:")
        for key, value in results.items():
            if key not in ["episode_rewards", "metrics"]:  #
                print(f"  {key}: {value}")
        print("=" * 70 + "\n")

    return results, algorithm


if __name__ == "__main__":
    import yaml

    # 加载配置
    with open("configs/network_dismantling/dqn.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 开始训练
    results, algo = train_from_cfg(config, verbose=True)
