"""
训练入口脚本
通过命令行参数执行指定的配置文件进行训练
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_dir: str = None, verbose: bool = True):
    """配置日志系统

    Args:
        log_dir: 日志保存目录，如果为 None 则不保存日志文件
        verbose: 是否在控制台输出日志
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []

    # 控制台日志
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)

    # 文件日志
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    # 设置根日志记录器
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 (日志目录: {log_dir if log_dir else 'None'})")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML/JSON/PY 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 根据文件类型加载不同的配置文件
    ext = os.path.splitext(config_path)[1].lower()

    if ext in (".yaml", ".yml"):
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f.read())
    elif ext in (".json", ".js"):
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif ext in (".py", ".pyc"):
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        raise ValueError(f"不支持的配置文件类型: {ext}")

    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CentriLearn 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置训练
  python tools/train.py configs/network_dismantling/dqn.yaml
  
  # 启用日志保存
  python tools/train.py configs/network_dismantling/dqn.yaml --use_logging --log_dir ./logs/train
  
  # 指定模型保存目录
  python tools/train.py configs/network_dismantling/dqn.yaml --ckpt_dir ./checkpoints
  
  # 从指定 checkpoint 恢复训练
  python tools/train.py configs/network_dismantling/dqn.yaml --resume ./checkpoints/model_best.pth
  
  # 禁用控制台输出
  python tools/train.py configs/network_dismantling/dqn.yaml --no_verbose
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
        default="./logs/train",
        help="日志保存目录 (默认: ./logs/train)",
    )
    parser.add_argument("--no_verbose", action="store_true", help="禁用控制台输出")

    # 训练相关
    parser.add_argument("--num_episodes", type=int, default=None, help="训练回合数")
    parser.add_argument("--max_steps", type=int, default=None, help="最大步数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument(
        "--is_valid", action="store_true", help="是否进行验证 (默认: False)"
    )

    # 优化相关
    parser.add_argument(
        "--benchmark", action="store_false", help="启用 PyTorch 基准测试"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="启用确定性模式"
    )
    parser.add_argument(
        "--memory_efficient", action="store_true", help="启用内存高效模式"
    )

    # Checkpoint 相关
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="模型保存目录 (默认: ./checkpoints)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="从指定 checkpoint 恢复训练"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="保存模型间隔，单位：episode (默认: DQN=100, PPO=50)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 1. 设置日志
    log_dir = args.log_dir if args.use_logging else None
    verbose = not args.no_verbose
    logger = setup_logging(log_dir, verbose)

    # 2. 加载配置
    if verbose:
        print("\n" + "=" * 70)
        print(f"加载配置文件: {args.config}")
        print("=" * 70)

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)

    # 3. 更新配置（命令行参数覆盖配置文件）
    update_dict = {}
    if args.num_episodes is not None:
        update_dict["num_episodes"] = args.num_episodes
    if args.max_steps is not None:
        update_dict["max_steps"] = args.max_steps
    if args.batch_size is not None:
        update_dict["batch_size"] = args.batch_size
    if args.save_interval is not None:
        update_dict["save_interval"] = args.save_interval

    update_dict["use_eval"] = args.is_valid

    # 添加 checkpoint 相关配置
    config["training"].update({"ckpt_dir": args.ckpt_dir, "resume": args.resume})
    if update_dict:
        config["training"].update(update_dict)
        logger.info(f"命令行参数覆盖: {update_dict}")

    # 4. 设置性能优化
    if verbose:
        print(f"\n[性能优化] 启用 PyTorch 性能优化配置")

    try:
        from centrilearn.utils.performance import \
            setup_performance_optimizations

        setup_performance_optimizations(
            device=config["algorithm"].get("device", "cuda"),
            benchmark=args.benchmark,
            deterministic=args.deterministic,
            memory_efficient=args.memory_efficient,
        )
    except ImportError:
        if verbose:
            print("  警告: 性能优化模块不可用，使用默认配置")

    # 5. 开始训练
    from centrilearn.utils.train import train_from_cfg

    try:
        results, algorithm = train_from_cfg(config, verbose=verbose)

        # 保存最终模型
        final_ckpt_path = os.path.join(args.ckpt_dir, "model_final.pth")
        algorithm.save_checkpoint(
            final_ckpt_path,
            episode=config["training"].get("num_episodes", 0),
            total_reward=results.get("total_reward", 0),
            episode_rewards=results.get("episode_rewards", []),
            best_reward=results.get("best_reward", float("-inf")),
        )
        logger.info(f"最终模型已保存: {final_ckpt_path}")

        logger.info("训练成功完成！")
        if verbose:
            print("\n训练结果摘要:")
            for key, value in results.items():
                if key != "episode_rewards":
                    print(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.warning("训练被用户中断 (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
