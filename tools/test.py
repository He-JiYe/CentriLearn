"""
测试入口脚本
通过命令行参数执行指定的配置文件进行测试
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from utils import load_config, setup_logging

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CentriLearn 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置测试
  python tools/test.py configs/network_dismantling/FINDER.yaml

  # 启用日志保存
  python tools/test.py configs/network_dismantling/FINDER.yaml --use_logging --log_dir ./logs

  # 指定数据目录
  python tools/test.py configs/network_dismantling/FINDER.yaml --data_dir data/small

  # 指定检查点路径
  python tools/test.py configs/network_dismantling/FINDER.yaml --checkpoint ./checkpoints/model_best.pth

  # 指定结果保存路径
  python tools/test.py configs/network_dismantling/FINDER.yaml --output_csv results/test.csv

  # 同时指定多个参数
  python tools/test.py configs/network_dismantling/FINDER.yaml --checkpoint ./checkpoints/model_best.pth --data_dir data/small --output_csv results/test.csv
        """,
    )

    parser.add_argument("config", type=str, help="配置文件路径")

    # 日志相关
    parser.add_argument(
        "--use_logging", action="store_true", help="是否启用日志记录到文件"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志保存目录",
    )
    parser.add_argument("--verbose", action="store_true", help="是否打印输出")

    # 测试相关
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型检查点路径（如果不指定则使用随机初始化的模型）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="测试数据目录路径",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="测试结果保存的 CSV 文件路径",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（用于结果复现）"
    )

    # 优化相关
    parser.add_argument(
        "--benchmark", action="store_false", help="启用 PyTorch 基准测试"
    )
    parser.add_argument("--deterministic", action="store_true", help="启用确定性模式")
    parser.add_argument(
        "--memory_efficient", action="store_true", help="启用内存高效模式"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    config_path = os.path.splitext(os.path.relpath(args.config, "configs"))[0]

    # 1. 设置日志
    log_dir = args.log_dir
    if args.log_dir is None:
        log_dir = (
            os.path.join("logs", config_path + "_test.log")
            if args.use_logging
            else None
        )
    verbose = args.verbose
    logger = setup_logging(log_dir)

    # 2. 加载配置
    if verbose:
        print("\n" + "=" * 70)
        print(f"加载配置文件: {args.config}")
        print("=" * 70)
    if logger:
        logger.info("=" * 70)
        logger.info(f"加载配置文件: {args.config}")
        logger.info("=" * 70)

    try:
        config = load_config(args.config)
    except Exception as e:
        if verbose:
            print(f"加载配置文件失败: {e}")
        if logger:
            logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)

    # 3. 命令行参数 > 配置文件 > 默认值
    update_dict = {}
    if args.checkpoint is not None:
        update_dict["checkpoint"] = args.checkpoint
    elif config["test"].get("checkpoint", None) is None:
        update_dict["checkpoint"] = os.path.join(
            "checkpoints", config_path, "checkpoint_final.pt"
        )
    if args.data_dir is not None:
        update_dict["data_dir"] = args.data_dir
    if args.output_csv is not None:
        update_dict["output_csv"] = args.output_csv
    elif config["test"].get("output_csv") is None:
        update_dict["output_csv"] = os.path.join("results", config_path + ".csv")
    if args.seed is not None:
        update_dict["seed"] = args.seed

    # 确保 test 配置存在
    if "test" not in config:
        config["test"] = {}

    if update_dict:
        config["test"] = {**config["test"], **update_dict}
        if verbose:
            print(f"命令行参数覆盖: {update_dict}")
        if logger:
            logger.info(f"命令行参数覆盖: {update_dict}")

    # 4. 设置随机种子
    seed = config["test"].get("seed", None)
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            args.deterministic = True
            args.memory_efficient = False

        if verbose:
            print(f"\n[随机种子] 设置随机种子: {seed}")
            print(f"  - Python random: {seed}")
            print(f"  - NumPy random: {seed}")
            print(f"  - PyTorch random: {seed}")
            if torch.cuda.is_available():
                print(f"  - CUDA random: {seed}")
                print(f"  - cuDNN deterministic: True")
        if logger:
            logger.info(f"设置随机种子: {seed}")

    # 5. 开始测试
    from centrilearn.utils.test import test_from_cfg

    # 确保结果目录存在
    output_csv = config["test"].get("output_csv")
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    try:
        test_from_cfg(
            config,
            verbose=verbose,
            logger=logger,
        )

    except KeyboardInterrupt:
        if logger:
            logger.warning("测试被用户中断 (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
