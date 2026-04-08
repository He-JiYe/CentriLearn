import logging
import os
from typing import Any, Dict


def setup_logging(log_path: str = None):
    """配置日志系统

    Args:
        log_path: 日志保存路径，如果为 None 则不保存日志文件
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []
    # 文件日志
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    # 设置根日志记录器
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 (日志目录: {log_path if log_path else 'None'})")
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
