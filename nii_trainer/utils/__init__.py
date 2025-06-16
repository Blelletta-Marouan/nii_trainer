"""
Utilities module for NII-Trainer.
"""

from .io import (
    save_checkpoint,
    load_checkpoint,
    save_predictions,
    load_predictions,
    create_directory,
    get_file_size,
    copy_file,
    move_file,
    cleanup_directory
)

from .medical import (
    get_image_info,
    resample_image,
    crop_to_nonzero,
    pad_image,
    compute_bounding_box,
    apply_window_level,
    normalize_hu_values,
    convert_coordinates,
    get_image_orientation,
    compute_volume_stats
)

from .device import (
    get_device,
    get_gpu_info,
    set_device,
    enable_mixed_precision,
    get_memory_usage,
    clear_gpu_cache,
    setup_distributed,
    is_distributed,
    get_world_size,
    get_rank
)

from .memory import (
    optimize_memory,
    get_memory_info,
    profile_memory,
    enable_memory_mapping,
    create_memory_pool,
    garbage_collect,
    monitor_memory_usage,
    MemoryProfiler
)

from .logging import (
    setup_logging,
    get_logger,
    log_system_info,
    log_model_info,
    log_experiment_config,
    create_tensorboard_logger,
    create_wandb_logger,
    LoggerManager
)

from .debugging import (
    profile_function,
    debug_tensor,
    check_gradients,
    visualize_activations,
    print_model_summary,
    check_memory_leak,
    benchmark_model,
    DebugHook,
    GradientMonitor
)

__all__ = [
    # I/O utilities
    "save_checkpoint", "load_checkpoint", "save_predictions", "load_predictions",
    "create_directory", "get_file_size", "copy_file", "move_file", "cleanup_directory",
    
    # Medical imaging utilities
    "get_image_info", "resample_image", "crop_to_nonzero", "pad_image",
    "compute_bounding_box", "apply_window_level", "normalize_hu_values",
    "convert_coordinates", "get_image_orientation", "compute_volume_stats",
    
    # Device management
    "get_device", "get_gpu_info", "set_device", "enable_mixed_precision",
    "get_memory_usage", "clear_gpu_cache", "setup_distributed", "is_distributed",
    "get_world_size", "get_rank",
    
    # Memory optimization
    "optimize_memory", "get_memory_info", "profile_memory", "enable_memory_mapping",
    "create_memory_pool", "garbage_collect", "monitor_memory_usage", "MemoryProfiler",
    
    # Logging
    "setup_logging", "get_logger", "log_system_info", "log_model_info",
    "log_experiment_config", "create_tensorboard_logger", "create_wandb_logger",
    "LoggerManager",
    
    # Debugging
    "profile_function", "debug_tensor", "check_gradients", "visualize_activations",
    "print_model_summary", "check_memory_leak", "benchmark_model", "DebugHook",
    "GradientMonitor"
]