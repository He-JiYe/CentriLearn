"""
训练入口函数
支持通过配置文件自动构建环境、算法、模型并进行训练
"""

from typing import Any, Dict, Tuple


def train_from_cfg(
    config: Dict[str, Any],
    verbose: bool = True,
    logger=None,
    tensorboard_writer=None,
    **kwargs,
) -> Tuple[Dict[str, Any], Any]:
    """从配置文件进行训练

    这是整个框架的统一训练入口。
    支持通过配置文件自动构建环境、算法、模型、优化器、调度器等组件。
    训练循环由 train.py 中的函数统一实现，不依赖算法实例的方法。

    Args:
        config: 训练配置字典，包含以下键：
            - algorithm: 算法配置
            - environment: 环境配置
            - training: 训练参数
        verbose: 是否打印训练日志
        logger: 日志记录器对象，如果为 None 则不记录日志
        tensorboard_writer: TensorBoard writer 对象，如果为 None 则不记录到 TensorBoard
        **kwargs: 额外的训练参数，覆盖 config['training'] 中的配置

    Returns:
        results: 训练结果字典，包含训练指标等信息
        algorithm: 训练完成的算法实例
    """
    # 检查参数
    if not isinstance(config, dict):
        if logger:
            logger.error(f"config must be a dict, but got {type(config)}")
        raise TypeError(f"config must be a dict, but got {type(config)}")

    required_keys = ["algorithm", "environment", "training"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        if logger:
            logger.error(
                f"config must contain {missing_keys}, but got keys: {config.keys()}"
            )
        raise KeyError(
            f"config must contain {missing_keys}, but got keys: {config.keys()}"
        )

    algorithm_required_keys = [
        "type",
        "model_cfg",
        "optimizer_cfg",
        "replaybuffer_cfg",
        "algo_cfg",
        "device",
    ]
    missing_algo_keys = [
        key for key in algorithm_required_keys if key not in config["algorithm"]
    ]
    if missing_algo_keys:
        if logger:
            logger.error(
                f'algorithm config must contain {missing_algo_keys}, but got keys: {config["algorithm"].keys()}'
            )
        raise KeyError(
            f'algorithm config must contain {missing_algo_keys}, but got keys: {config["algorithm"].keys()}'
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
    if logger:
        logger.info(f"开始训练 - 算法类型: {algorithm_cfg['type']}")

    # 1. 构建环境
    if verbose:
        print(f"\n[1/4] 构建环境: {env_cfg.get('type', 'unknown')}")
    if logger:
        logger.info(f"构建环境: {env_cfg.get('type', 'unknown')}")

    # 延迟导入构建函数
    from centrilearn.utils.builder import build_environment

    env = build_environment(env_cfg)
    if verbose:
        print(f"      [OK] 环境构建完成: {env}")
    if logger:
        logger.info(f"环境构建完成: {env}")

    # 3. 构建算法
    if verbose:
        print(f"\n[2/4] 构建算法: {algorithm_cfg['type']}")
        print(f"      - 模型类型: {algorithm_cfg.get('model', 'unknown')}")
        print(
            f"      - 优化器: {algorithm_cfg.get('optimizer_cfg', {}).get('type')} (lr={algorithm_cfg.get('optimizer_cfg', {}).get('lr', 'N/A')})"
        )
    if logger:
        logger.info(f"构建算法: {algorithm_cfg['type']}")
        logger.info(f"模型类型: {algorithm_cfg.get('model', 'unknown')}")
        logger.info(
            f"优化器: {algorithm_cfg.get('optimizer_cfg', {}).get('type')} (lr={algorithm_cfg.get('optimizer_cfg', {}).get('lr', 'N/A')})"
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

    algorithm = build_algorithm(algorithm_cfg)

    if verbose:
        print(f"      [OK] 算法构建完成: {algorithm}")
    if logger:
        logger.info(f"算法构建完成: {algorithm}")

    # 4. 检查是否需要恢复训练
    resume_from = training_cfg.get("resume")
    if resume_from:
        if verbose:
            print(f"\n 恢复训练...")
            print(f"      从检查点恢复: {resume_from}")
        if logger:
            logger.info(f"恢复训练，从检查点恢复: {resume_from}")

        # 加载检查点
        try:
            checkpoint = algorithm.load_checkpoint(resume_from)
            if verbose:
                print(f"      [OK] 检查点加载成功")
                print(f"      训练步数: {algorithm.training_step}")
                if "episode" in checkpoint:
                    print(f"      恢复episode: {checkpoint['episode']}")
            if logger:
                logger.info(f"检查点加载成功，训练步数: {algorithm.training_step}")
                if "episode" in checkpoint:
                    logger.info(f"恢复episode: {checkpoint['episode']}")
        except Exception as e:
            if verbose:
                print(f"      [警告] 检查点加载失败: {e}")
                print(f"      将从头开始训练")
            if logger:
                logger.warning(f"检查点加载失败: {e}，将从头开始训练")

    # 5. 执行训练
    final_training_cfg = {**training_cfg, **kwargs}

    if verbose:
        if not resume_from:
            print(f"\n[3/4] 开始训练...")
        else:
            print(f"\n[3/4] 继续训练...")
        print(f"      训练配置: {final_training_cfg}")
    if logger:
        if not resume_from:
            logger.info("开始训练...")
        else:
            logger.info("继续训练...")
        logger.info(f"训练配置: {final_training_cfg}")

    results = algorithm.learn(
        env,
        final_training_cfg,
        verbose=verbose,
        logger=logger,
        tensorboard_writer=tensorboard_writer,
    )

    # 6. 训练完成
    if verbose:
        print(f"\n[4/4] 训练完成！")
        print(f"\n训练结果:")
        for key, value in results.items():
            if key not in ["episode_rewards", "metrics"]:  #
                print(f"  {key}: {value}")
        print("=" * 70 + "\n")
    if logger:
        logger.info("训练完成！")
        logger.info(f"训练结果: {results}")
        for key, value in results.items():
            if key not in ["episode_rewards", "metrics"]:  #
                logger.info(f"  {key}: {value}")

    return results, algorithm
