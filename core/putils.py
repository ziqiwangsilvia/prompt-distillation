import torch

def print_gpu_utilization() -> list:
    """Print and return GPU memory usage in MB (using pyrsmi/rocml)."""
    from pyrsmi import rocml
    rocml.smi_initialize()
    ndevices = rocml.smi_get_device_count()
    used = [rocml.smi_get_device_memory_used(i) for i in range(ndevices)]
    usage_str = "GPU memory used (MB): " + " ".join(f"[{i}]:{used[i]//1024**2}" for i in range(ndevices))
    print(usage_str)
    rocml.smi_shutdown()
    return used

def print_cuda_memory_utilization(rank: int = 0) -> None:
    """Print CUDA memory allocation in MB for the specified device."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    used = torch.cuda.memory_allocated(rank)
    print(f"CUDA[{rank}] memory allocated: {used//1024**2} MB.")

