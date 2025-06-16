"""
Debugging utilities for NII-Trainer.
"""

import time
import torch
import functools
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from ..core.exceptions import NIITrainerError


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def debug_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Debug tensor information."""
    print(f"=== Debug: {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Min: {tensor.min().item():.6f}")
    print(f"Max: {tensor.max().item():.6f}")
    print(f"Mean: {tensor.mean().item():.6f}")
    print(f"Std: {tensor.std().item():.6f}")
    print(f"Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"Has Inf: {torch.isinf(tensor).any().item()}")
    print("=" * (len(name) + 12))


def check_gradients(model: torch.nn.Module, threshold: float = 1e-6) -> Dict[str, Any]:
    """Check model gradients for debugging."""
    gradient_info = {
        'total_params': 0,
        'params_with_grad': 0,
        'zero_gradients': 0,
        'nan_gradients': 0,
        'inf_gradients': 0,
        'gradient_norms': {},
        'max_gradient': 0.0,
        'min_gradient': float('inf')
    }
    
    for name, param in model.named_parameters():
        gradient_info['total_params'] += 1
        
        if param.grad is not None:
            gradient_info['params_with_grad'] += 1
            grad_norm = param.grad.norm().item()
            gradient_info['gradient_norms'][name] = grad_norm
            
            gradient_info['max_gradient'] = max(gradient_info['max_gradient'], grad_norm)
            gradient_info['min_gradient'] = min(gradient_info['min_gradient'], grad_norm)
            
            if grad_norm < threshold:
                gradient_info['zero_gradients'] += 1
            
            if torch.isnan(param.grad).any():
                gradient_info['nan_gradients'] += 1
                print(f"Warning: NaN gradients in {name}")
            
            if torch.isinf(param.grad).any():
                gradient_info['inf_gradients'] += 1
                print(f"Warning: Inf gradients in {name}")
    
    return gradient_info


def visualize_activations(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Visualize model activations for debugging."""
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    if layer_names is None:
        # Register hooks for all modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn(name)))
    else:
        # Register hooks for specified layers
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
    
    try:
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Print activation statistics
        print("=== Activation Statistics ===")
        for name, activation in activations.items():
            print(f"{name}:")
            print(f"  Shape: {activation.shape}")
            print(f"  Mean: {activation.mean().item():.6f}")
            print(f"  Std: {activation.std().item():.6f}")
            print(f"  Min: {activation.min().item():.6f}")
            print(f"  Max: {activation.max().item():.6f}")
            print(f"  Sparsity: {(activation == 0).float().mean().item():.3f}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return activations


def print_model_summary(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> None:
    """Print detailed model summary."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size, device=device)
    
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer':<30} {'Output Shape':<20} {'Param #':<15}")
    print("-" * 80)
    
    def hook_fn(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)
        
        m_key = f"{class_name}-{module_idx+1}"
        summary[m_key] = {
            'input_shape': list(input[0].size()) if input else [],
            'output_shape': list(output.size()) if hasattr(output, 'size') else [],
            'nb_params': sum([param.nelement() for param in module.parameters()])
        }
    
    summary = {}
    hooks = []
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))
    
    try:
        with torch.no_grad():
            model(dummy_input)
        
        for layer_name, layer_info in summary.items():
            output_shape = str(layer_info['output_shape'])
            num_params = layer_info['nb_params']
            
            print(f"{layer_name:<30} {output_shape:<20} {num_params:<15,}")
            total_params += num_params
    
    finally:
        for hook in hooks:
            hook.remove()
    
    # Count trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 80)


def check_memory_leak(
    func: Callable,
    iterations: int = 5,
    threshold_mb: float = 100.0
) -> bool:
    """Check for memory leaks in a function."""
    import gc
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i in range(iterations):
        func()
        if i == 0:
            # Memory after first iteration (baseline)
            baseline_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Check for memory growth
    memory_growth = (final_memory - baseline_memory) / 1024**2  # MB
    
    if memory_growth > threshold_mb:
        print(f"Warning: Potential memory leak detected!")
        print(f"Memory growth: {memory_growth:.2f} MB over {iterations-1} iterations")
        return True
    
    return False


def benchmark_model(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Benchmark model inference performance."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    dummy_input = torch.randn(1, *input_size, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'fps': 1.0 / float(np.mean(times)),
        'throughput': num_runs / float(np.sum(times))
    }


class DebugHook:
    """Debug hook for monitoring layer outputs during training."""
    
    def __init__(self, module: torch.nn.Module, name: str):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activations = []
    
    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'has_nan': torch.isnan(output).any().item(),
                'has_inf': torch.isinf(output).any().item()
            }
            self.activations.append(stats)
    
    def remove(self):
        self.hook.remove()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.activations:
            return {}
        
        means = [a['mean'] for a in self.activations]
        stds = [a['std'] for a in self.activations]
        
        return {
            'layer_name': self.name,
            'num_calls': len(self.activations),
            'mean_activation': np.mean(means),
            'std_activation': np.mean(stds),
            'has_anomalies': any(a['has_nan'] or a['has_inf'] for a in self.activations)
        }


class GradientMonitor:
    """Monitor gradients during training for debugging."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradient_history = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self._gradient_hook(grad, name))
                self.hooks.append(hook)
    
    def _gradient_hook(self, grad: torch.Tensor, param_name: str):
        if grad is not None:
            grad_norm = grad.norm().item()
            self.gradient_history.append({
                'param_name': param_name,
                'grad_norm': grad_norm,
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item()
            })
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        if not self.gradient_history:
            return {}
        
        grad_norms = [g['grad_norm'] for g in self.gradient_history]
        
        return {
            'num_gradients': len(self.gradient_history),
            'mean_grad_norm': np.mean(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'min_grad_norm': np.min(grad_norms),
            'nan_gradients': sum(g['has_nan'] for g in self.gradient_history),
            'inf_gradients': sum(g['has_inf'] for g in self.gradient_history)
        }
    
    def clear_history(self):
        self.gradient_history.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()