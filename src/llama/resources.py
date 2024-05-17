import torch
import pynvml


def print_gpu_utilization():
    """Prints the GPU memory occupied by the current process."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**3} GB.")
