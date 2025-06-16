"""
Device management utilities for NII-Trainer.
"""

import os
import torch
import platform
from typing import Dict, List, Optional, Tuple, Any
from ..core.exceptions import NIITrainerError


def get_device(device_id: Optional[str] = None) -> torch.device:
    """Get the best available device for computation."""
    if device_id == "cpu":
        return torch.device("cpu")
    elif device_id and device_id.startswith("cuda"):
        if not torch.cuda.is_available():
            raise NIITrainerError(f"CUDA device {device_id} requested but CUDA not available")
        return torch.device(device_id)
    elif device_id == "auto" or device_id is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_id)


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {"available": False, "devices": []}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "id": i,
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
            "memory_allocated": torch.cuda.memory_allocated(i),
            "memory_reserved": torch.cuda.memory_reserved(i),
            "memory_free": props.total_memory - torch.cuda.memory_reserved(i)
        }
        gpu_info["devices"].append(device_info)
    
    return gpu_info


def set_device(device: torch.device) -> None:
    """Set the current device."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise NIITrainerError("CUDA not available")
        torch.cuda.set_device(device)


def enable_mixed_precision() -> bool:
    """Enable mixed precision training if supported."""
    if torch.cuda.is_available():
        # Check if GPU supports mixed precision
        if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
            return True
    return False


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, int]:
    """Get current memory usage for device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device)
        }
    else:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percentage": memory.percent
        }


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: int = 1,
    rank: int = 0
) -> bool:
    """Setup distributed training."""
    if not torch.distributed.is_available():
        return False
    
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        return True
    except Exception as e:
        raise NIITrainerError(f"Failed to setup distributed training: {e}")


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    """Get world size for distributed training."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank for distributed training."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def optimize_torch_settings() -> None:
    """Apply optimizations for PyTorch performance."""
    # Set number of threads for CPU operations
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(min(8, torch.get_num_threads()))
    
    # Enable cuDNN benchmark mode for consistent input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set float32 matmul precision for better performance on newer GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available
    }
    
    if torch.cuda.is_available():
        system_info["gpu_info"] = get_gpu_info()
    
    return system_info