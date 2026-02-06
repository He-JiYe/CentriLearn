"""
PyTorch 性能优化配置和工具
包含各种性能优化设置和最佳实践
"""

from typing import Optional

import torch
import torch.backends.cudnn as cudnn


def setup_performance_optimizations(
    device: str = "cuda",
    benchmark: bool = True,
    deterministic: bool = False,
    memory_efficient: bool = True,
    enable_cuda_graphs: bool = False,
) -> None:
    """设置 PyTorch 性能优化配置

    Args:
        device: 运行设备 ('cuda' 或 'cpu')
        benchmark: 是否启用 cudnn 基准测试
        deterministic: 是否启用确定性计算
        memory_efficient: 是否启用内存优化
        enable_cuda_graphs: 是否启用 CUDA 图优化
    """

    # CUDA 相关优化
    if device == "cuda" and torch.cuda.is_available():
        # 启用 cudnn 基准测试（对于固定输入大小性能更好）
        cudnn.benchmark = benchmark

        # 设置确定性计算（如果需要可重现结果）
        cudnn.deterministic = deterministic
        torch.backends.cudnn.deterministic = deterministic

        # 启用内存优化
        if memory_efficient:
            # 启用内存高效注意力（如果可用）
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        # CUDA 图优化（适用于固定计算图）
        if enable_cuda_graphs:
            torch.backends.cuda.enable_cudagraphs()

    # 通用优化
    # 启用更快的矩阵乘法（如果可用）
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"性能优化已启用: benchmark={benchmark}, deterministic={deterministic}")


def optimize_memory_usage():
    """优化内存使用"""

    if torch.cuda.is_available():
        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.9)  # 保留 10% 内存给系统


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """启用梯度检查点（用于大模型的内存优化）

    Args:
        model: 需要启用梯度检查点的模型
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "apply"):
        # 递归应用梯度检查点
        def enable_checkpointing(module):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True

        model.apply(enable_checkpointing)


def get_memory_info(device: Optional[str] = None) -> dict:
    """获取内存使用信息

    Args:
        device: 设备名称

    Returns:
        内存信息字典
    """
    memory_info = {}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        memory_info.update(
            {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
                "max_reserved": torch.cuda.max_memory_reserved() / 1024**3,  # GB
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
            }
        )

    # 系统内存信息
    import psutil

    memory_info.update(
        {
            "system_used": psutil.virtual_memory().used / 1024**3,  # GB
            "system_total": psutil.virtual_memory().total / 1024**3,  # GB
            "system_available": psutil.virtual_memory().available / 1024**3,  # GB
        }
    )

    return memory_info


def print_memory_stats(prefix: str = "") -> None:
    """打印内存统计信息

    Args:
        prefix: 前缀字符串
    """
    memory_info = get_memory_info()

    print(f"{prefix}内存使用统计:")
    if "allocated" in memory_info:
        print(f"  GPU 已分配: {memory_info['allocated']:.2f} GB")
        print(f"  GPU 已保留: {memory_info['reserved']:.2f} GB")
        print(f"  GPU 最大分配: {memory_info['max_allocated']:.2f} GB")

    print(f"  系统已使用: {memory_info['system_used']:.2f} GB")
    print(f"  系统可用: {memory_info['system_available']:.2f} GB")


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.memory_stats = []
        self.timings = {}

    def start_timing(self, name: str):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[name] = {
            "start": (
                torch.cuda.Event(enable_timing=True)
                if torch.cuda.is_available()
                else None
            ),
            "end": None,
            "start_time": (
                torch.cuda.Event(enable_timing=True)
                if torch.cuda.is_available()
                else None
            ),
        }
        if torch.cuda.is_available():
            self.timings[name]["start"].record()

    def end_timing(self, name: str):
        """结束计时"""
        if name in self.timings:
            if torch.cuda.is_available():
                self.timings[name]["end"] = torch.cuda.Event(enable_timing=True)
                self.timings[name]["end"].record()
                torch.cuda.synchronize()
                elapsed_time = self.timings[name]["start"].elapsed_time(
                    self.timings[name]["end"]
                )
                self.timings[name]["elapsed_ms"] = elapsed_time

    def get_timing(self, name: str) -> float:
        """获取计时结果（毫秒）"""
        if name in self.timings and "elapsed_ms" in self.timings[name]:
            return self.timings[name]["elapsed_ms"]
        return 0.0

    def record_memory(self):
        """记录内存使用"""
        self.memory_stats.append(get_memory_info())

    def print_summary(self):
        """打印性能摘要"""
        print("\n=== 性能监控摘要 ===")

        # 打印计时信息
        for name, timing in self.timings.items():
            if "elapsed_ms" in timing:
                print(f"{name}: {timing['elapsed_ms']:.2f} ms")

        # 打印内存信息
        if self.memory_stats:
            latest = self.memory_stats[-1]
            print(f"\n最新内存使用:")
            if "allocated" in latest:
                print(f"  GPU 分配: {latest['allocated']:.2f} GB")
            print(f"  系统使用: {latest['system_used']:.2f} GB")


# 默认性能配置
default_performance_config = {
    "benchmark": True,
    "deterministic": False,
    "memory_efficient": True,
    "enable_cuda_graphs": False,
}


if __name__ == "__main__":
    # 测试性能优化设置
    setup_performance_optimizations()
    print_memory_stats("初始化后")

    # 创建性能监控器
    monitor = PerformanceMonitor()
    monitor.start_timing("test_operation")

    # 模拟一些操作
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    result = torch.matmul(x, y)

    monitor.end_timing("test_operation")
    monitor.record_memory()
    monitor.print_summary()
