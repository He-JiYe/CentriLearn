"""
训练入口脚本
通过命令行参数执行指定的配置文件进行训练
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
        description="CentriLearn 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置训练
  python tools/train.py configs/network_dismantling/dqn.yaml
  
  # 启用日志保存
  python tools/train.py configs/network_dismantling/dqn.yaml --use_logging --log_dir ./logs
  
  # 启用 TensorBoard
  python tools/train.py configs/network_dismantling/dqn.yaml --use_tensorboard --tensorboard_dir ./runs
  
  # 同时启用日志和 TensorBoard
  python tools/train.py configs/network_dismantling/dqn.yaml --use_logging --use_tensorboard
  
  # 指定模型保存目录
  python tools/train.py configs/network_dismantling/dqn.yaml --ckpt_dir ./checkpoints
  
  # 从指定 checkpoint 恢复训练
  python tools/train.py configs/network_dismantling/dqn.yaml --resume ./checkpoints/model_best.pth
        """,
    )

    parser.add_argument("config", type=str, help="配置文件路径")

    # 日志相关
    parser.add_argument(
        "--use_logging", action="store_true", help="是否启用日志记录到文件"
    )
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志保存目录",
    )
    parser.add_argument("--verbose", action="store_true", help="是否打印输出")
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="是否启用 TensorBoard 记录"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard 保存目录",
    )

    # 训练相关
    parser.add_argument("--num_episodes", type=int, default=None, help="训练回合数")
    parser.add_argument("--max_steps", type=int, default=None, help="最大步数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（用于结果复现）"
    )

    # 优化相关
    parser.add_argument(
        "--benchmark", action="store_true", help="启用 PyTorch 基准测试"
    )
    parser.add_argument("--deterministic", action="store_true", help="启用确定性模式")
    parser.add_argument(
        "--memory_efficient", action="store_true", help="启用内存高效模式"
    )

    # Checkpoint 相关
    parser.add_argument("--save_path", type=str, default=None, help="模型保存目录")
    parser.add_argument(
        "--resume", type=str, default=None, help="从指定 checkpoint 恢复训练"
    )
    parser.add_argument(
        "--save_interval", type=int, default=None, help="保存模型间隔，单位：episode "
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
            os.path.join("logs", config_path + "_train.log")
            if args.use_logging
            else None
        )
    verbose = args.verbose
    logger = setup_logging(log_dir)

    # 设置 TensorBoard writer
    tensorboard_writer = None
    if args.use_tensorboard:
        tensorboard_dir = args.tensorboard_dir
        if tensorboard_dir is None:
            tensorboard_dir = os.path.join("runs", config_path)

        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_writer = SummaryWriter(tensorboard_dir)
            logger.info(f"TensorBoard 已启用，保存目录: {tensorboard_dir}")
        except ImportError:
            logger.warning(
                "TensorBoard 不可用，请安装 tensorboard: pip install tensorboard"
            )
        except Exception as e:
            logger.warning(f"TensorBoard 初始化失败: {e}")

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
    if args.num_episodes is not None:
        update_dict["num_episodes"] = args.num_episodes
    if args.max_steps is not None:
        update_dict["max_steps"] = args.max_steps
    if args.batch_size is not None:
        update_dict["batch_size"] = args.batch_size
    if args.log_interval is not None:
        update_dict["log_interval"] = args.log_interval
    if args.save_interval is not None:
        update_dict["save_interval"] = args.save_interval
    if args.save_path is not None:
        update_dict["save_path"] = args.save_path
    elif config["training"].get("save_path", None) is None:
        update_dict["save_path"] = os.path.join("checkpoints", config_path)
    if args.resume is not None:
        update_dict["resume"] = args.resume

    if args.seed is not None:
        update_dict["seed"] = args.seed

    if update_dict:
        config["training"] = {**config["training"], **update_dict}
        if verbose:
            print(f"命令行参数覆盖: {update_dict}")
        if logger:
            logger.info(f"命令行参数覆盖: {update_dict}")

    # 4. 设置随机种子
    seed = config["training"].get("seed", None)
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

    # 5. 开始训练
    from centrilearn.utils.train import train_from_cfg

    os.makedirs(config["training"].get("save_path"), exist_ok=True)

    try:
        train_from_cfg(
            config,
            verbose=verbose,
            logger=logger,
            tensorboard_writer=tensorboard_writer,
        )

    except KeyboardInterrupt:
        if logger:
            logger.warning("训练被用户中断 (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
            if logger:
                logger.info("TensorBoard writer 已关闭")


if __name__ == "__main__":
    main()
