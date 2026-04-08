"""
测试入口函数
支持通过配置文件自动构建环境、算法并进行测试
"""

import copy
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def test_from_cfg(
    config: Dict[str, Any],
    verbose: bool = True,
    logger=None,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Any]:
    """从配置文件进行测试

    这是整个框架的统一测试入口。支持通过配置文件自动构建环境、算法、模型等组件。
    可以对多个图进行测试，并将结果保存到 CSV 文件中。

    Args:
        config: 测试配置字典，包含以下键：
            - algorithm: 算法配置
            - environment: 环境配置（基础配置）
            - test: 测试配置, 包括 checkpoint, data_dir, output_csv。
        verbose: 是否打印测试日志
        logger: 日志记录器对象，如果为 None 则不记录日志
        **kwargs: 额外的测试参数，覆盖 config['test'] 中的配置

    Returns:
        results: 测试结果列表，每个元素是对应图的测试结果
    """
    # 检查参数
    if not isinstance(config, dict):
        if logger:
            logger.error(f"config must be a dict, but got {type(config)}")
        raise TypeError(f"config must be a dict, but got {type(config)}")

    required_keys = ["algorithm", "environment", "test"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        if logger:
            logger.error(
                f"config must contain {missing_keys}, but got keys: {config.keys()}"
            )
        raise KeyError(
            f"config must contain {missing_keys}, but got keys: {config.keys()}"
        )

    algorithm_cfg, env_cfg, test_cfg = (
        config["algorithm"],
        config["environment"],
        config["test"],
    )

    # 命令行参数覆盖配置文件参数
    final_test_cfg = {**test_cfg, **kwargs}

    if verbose:
        print("\n" + "=" * 70)
        print(f"开始测试 - 算法类型: {algorithm_cfg['type']}")
        print("=" * 70)
    if logger:
        logger.info(f"开始测试 - 算法类型: {algorithm_cfg['type']}")

    # 1. 构建算法
    if verbose:
        print(f"\n[1/3] 构建算法: {algorithm_cfg['type']}")
        print(f"      - 模型类型: {algorithm_cfg.get('model', 'unknown')}")
    if logger:
        logger.info(f"构建算法: {algorithm_cfg['type']}")
        logger.info(f"模型类型: {algorithm_cfg.get('model', 'unknown')}")

    from centrilearn.utils.builder import build_algorithm
    from centrilearn.utils.registry import ALGORITHMS

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

    # 加载检查点
    checkpoint_path = final_test_cfg.get("checkpoint")
    output_csv = final_test_cfg.get("output_csv")

    if checkpoint_path:
        if verbose:
            print(f"\n 加载检查点...")
            print(f"      从 {checkpoint_path} 加载模型")
        if logger:
            logger.info(f"从 {checkpoint_path} 加载模型")

        try:
            algorithm.load_checkpoint(checkpoint_path)
            if verbose:
                print(f"      [OK] 检查点加载成功")
                print(f"      训练步数: {algorithm.training_step}")
            if logger:
                logger.info(f"检查点加载成功，训练步数: {algorithm.training_step}")
        except Exception as e:
            if verbose:
                print(f"      [警告] 检查点加载失败: {e}")
                print(f"      将使用随机初始化的模型进行测试")
            if logger:
                logger.warning(f"检查点加载失败: {e}，将使用随机初始化的模型进行测试")

            base_name, ext = os.path.splitext(output_csv)
            output_csv = f"{base_name}_random{ext}"
    else:
        if verbose:
            print(f"      [警告] 未指定模型参数")
            print(f"      将使用随机初始化的模型进行测试")
        if logger:
            logger.warning(f"未指定模型参数，将使用随机初始化的模型进行测试")

        base_name, ext = os.path.splitext(output_csv)
        output_csv = f"{base_name}_random{ext}"

    # 2. 准备测试图
    if verbose:
        print(f"\n[2/3] 准备测试图...")
    if logger:
        logger.info("准备测试图...")

    test_envs = []
    data_dir = test_cfg.get("data_dir")
    if data_dir is None:
        if logger:
            logger.error("test_cfg 必须包含 'data_dir' 键")
        raise ValueError("test_cfg 必须包含 'data_dir' 键")

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent.parent / data_dir

    if not data_path.exists():
        if logger:
            logger.error(f"数据目录不存在: {data_path}")
        raise ValueError(f"数据目录不存在: {data_path}")

    # 遍历目录下的所有图文件
    for graph_file in sorted(data_path.glob("*.txt")):
        current_env_cfg = copy.deepcopy(env_cfg)
        current_env_cfg["graph_file"] = str(graph_file)
        test_envs.append((graph_file.stem, current_env_cfg))

    if not test_envs:
        if logger:
            logger.error(f"在目录 {data_path} 中未找到任何 .txt 文件")
        raise ValueError(f"在目录 {data_path} 中未找到任何 .txt 文件")

    if verbose:
        print(f"      [OK] 共准备 {len(test_envs)} 个测试环境")
        for i, (graph_name, env_cfg_item) in enumerate(test_envs[:5], 1):
            print(f"        {i}. {graph_name}")
        if len(test_envs) > 5:
            print(f"        ... 还有 {len(test_envs) - 5} 个")
    if logger:
        logger.info(f"共准备 {len(test_envs)} 个测试环境")

    # 3. 执行测试
    if verbose:
        print(f"\n[3/3] 开始测试...")
        print(f"      训练配置: {final_test_cfg}")
    if logger:
        logger.info("开始测试...")
        logger.info(f"训练配置: {final_test_cfg}")

    results = []
    output_path = None
    csv_fields = None

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    use_gcc = final_test_cfg.get("use_gcc", False)
    for i, (graph_name, env_cfg_item) in enumerate(test_envs):
        from centrilearn.utils.builder import build_environment

        # 构建环境
        env = build_environment(env_cfg_item)

        # 执行 rollout
        if verbose:
            print(f"      [{i+1} / {len(test_envs)}] {env} 开始测试")
        if logger:
            logger.info(f"开始测试 {env}")

        result = {"graph_name": graph_name}
        result.update(algorithm.rollout(env, use_gcc))

        if verbose:
            print(
                f"{graph_name} 完成测试，节点数量: {result['num_nodes']}, 花费时间: {result['rollout_time']:.2f}s"
            )
        if logger:
            logger.info(
                f"测试 {graph_name}, 节点数量: {result['num_nodes']}, 用时: {result['rollout_time']:.2f}s"
            )

        # 将结果追加到 CSV（第一次时写入表头）
        if output_path:
            with open(
                output_path, "a" if csv_fields else "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)

                # 第一次写入表头
                if csv_fields is None:
                    csv_fields = result.keys()
                    writer.writerow(csv_fields)

                # 动态处理不同类型的值
                row_values = []
                for key, value in result.items():
                    if isinstance(value, (list, tuple)):
                        row_values.append(",".join(map(str, value)))
                    else:
                        row_values.append(str(value))

                writer.writerow(row_values)

        results.append(result)

    # 测试完成
    if verbose:
        print(f"\n[完成] 测试结束！")
        print(f"  结果保存到: {output_csv if output_csv else '未保存'}")
        print("=" * 70 + "\n")
    if logger:
        logger.info("测试完成！")
        if output_csv:
            logger.info(f"结果保存到: {output_csv}")

    return results
