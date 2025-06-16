"""
Memory optimization utilities for NII-Trainer.
"""

import gc
import torch
import psutil
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from ..core.exceptions import NIITrainerError


def optimize_memory() -> None:
    """Apply general memory optimizations."""
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Set memory allocation strategy
    if hasattr(torch, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)


def get_memory_info() -> Dict[str, Any]:
    """Get comprehensive memory information."""
    info = {
        "system": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used,
            "percentage": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {}
        for i in range(torch.cuda.device_count()):
            info["gpu"][f"device_{i}"] = {
                "total": torch.cuda.get_device_properties(i).total_memory,
                "allocated": torch.cuda.memory_allocated(i),
                "reserved": torch.cuda.memory_reserved(i),
                "free": torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
            }
    
    return info


@contextmanager
def profile_memory(device: Optional[torch.device] = None):
    """Context manager for profiling memory usage."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Record initial state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
    else:
        start_memory = psutil.virtual_memory().used
    
    try:
        yield
    finally:
        # Record final state and print stats
        if device.type == "cuda":
            end_allocated = torch.cuda.memory_allocated(device)
            end_reserved = torch.cuda.memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            
            print(f"Memory Usage on {device}:")
            print(f"  Allocated: {(end_allocated - start_allocated) / 1024**2:.2f} MB")
            print(f"  Reserved: {(end_reserved - start_reserved) / 1024**2:.2f} MB")
            print(f"  Peak Allocated: {peak_allocated / 1024**2:.2f} MB")
            print(f"  Peak Reserved: {peak_reserved / 1024**2:.2f} MB")
        else:
            end_memory = psutil.virtual_memory().used
            print(f"System Memory Usage:")
            print(f"  Used: {(end_memory - start_memory) / 1024**2:.2f} MB")


def enable_memory_mapping(enable: bool = True) -> None:
    """Enable memory mapping for large datasets."""
    if enable:
        # Set environment variables for memory mapping
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def create_memory_pool(size_mb: int = 1024) -> None:
    """Create a memory pool for efficient allocation."""
    if torch.cuda.is_available():
        # Reserve memory pool
        pool_size = size_mb * 1024 * 1024
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Pre-allocate and free to establish pool
        dummy = torch.zeros((pool_size // 4,), dtype=torch.float32, device='cuda')
        del dummy
        torch.cuda.empty_cache()


def garbage_collect(force: bool = False) -> int:
    """Perform garbage collection and return number of objects collected."""
    if force:
        # Force collection of all generations
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
    else:
        collected = gc.collect()
    
    # Clear PyTorch cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return collected


class MemoryProfiler:
    """Advanced memory profiler for tracking usage over time."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.history: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.history.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            memory_info = get_memory_info()
            memory_info['timestamp'] = time.time()
            self.history.append(memory_info)
            time.sleep(self.interval)
    
    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak memory usage from monitoring history."""
        if not self.history:
            return {}
        
        peak_system = max(self.history, key=lambda x: x['system']['used'])
        result = {'system_peak': peak_system}
        
        if 'gpu' in self.history[0]:
            gpu_peaks = {}
            for device_name in self.history[0]['gpu'].keys():
                gpu_peaks[device_name] = max(
                    self.history,
                    key=lambda x: x['gpu'][device_name]['allocated']
                )
            result['gpu_peaks'] = gpu_peaks
        
        return result
    
    def save_report(self, filepath: str) -> None:
        """Save monitoring report to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'history': self.history,
                'peak_usage': self.get_peak_usage(),
                'monitoring_interval': self.interval
            }, f, indent=2)


def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler(interval=0.5)
        profiler.start_monitoring()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.stop_monitoring()
            peak_info = profiler.get_peak_usage()
            
            print(f"Memory usage for {func.__name__}:")
            if 'system_peak' in peak_info:
                peak_mb = peak_info['system_peak']['system']['used'] / 1024**2
                print(f"  Peak system memory: {peak_mb:.2f} MB")
            
            if 'gpu_peaks' in peak_info:
                for device, peak in peak_info['gpu_peaks'].items():
                    peak_mb = peak['gpu'][device]['allocated'] / 1024**2
                    print(f"  Peak GPU memory ({device}): {peak_mb:.2f} MB")
    
    return wrapper


def check_memory_leak(threshold_mb: float = 100.0) -> bool:
    """Check for potential memory leaks."""
    initial_memory = psutil.virtual_memory().used
    
    # Force garbage collection
    garbage_collect(force=True)
    
    # Wait a moment for cleanup
    time.sleep(0.1)
    
    final_memory = psutil.virtual_memory().used
    leaked_mb = (final_memory - initial_memory) / 1024**2
    
    if leaked_mb > threshold_mb:
        print(f"Warning: Potential memory leak detected! {leaked_mb:.2f} MB not freed")
        return True
    
    return False


def get_memory_efficient_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    max_memory_mb: float = 8000,
    device: Optional[torch.device] = None
) -> int:
    """Estimate optimal batch size based on available memory."""
    if device is None:
        device = next(model.parameters()).device
    
    if device.type != "cuda":
        return 32  # Default for CPU
    
    # Get available memory
    available_memory = torch.cuda.get_device_properties(device).total_memory
    available_mb = available_memory / 1024**2
    target_memory = min(max_memory_mb, available_mb * 0.8)  # Use 80% of available
    
    # Test with small batch to estimate memory per sample
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, *input_shape, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        try:
            _ = model(test_input)
            memory_per_sample = torch.cuda.max_memory_allocated(device) / 1024**2
        except RuntimeError:
            return 1  # Very limited memory
        
        del test_input
        torch.cuda.empty_cache()
    
    # Estimate batch size (with safety margin)
    if memory_per_sample > 0:
        estimated_batch_size = int(target_memory / memory_per_sample * 0.7)
        return max(1, min(estimated_batch_size, 128))  # Cap at reasonable maximum
    
    return 16  # Fallback default