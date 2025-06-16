"""
Logging utilities for NII-Trainer.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from ..core.exceptions import NIITrainerError


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """Setup logging configuration."""
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger('nii_trainer')
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'nii_trainer') -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log comprehensive system information."""
    if logger is None:
        logger = get_logger()
    
    from .device import get_system_info
    system_info = get_system_info()
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {system_info['platform']}")
    logger.info(f"Python Version: {system_info['python_version']}")
    logger.info(f"PyTorch Version: {system_info['pytorch_version']}")
    logger.info(f"CUDA Available: {system_info['cuda_available']}")
    
    if system_info['cuda_available']:
        logger.info(f"CUDA Version: {system_info['cuda_version']}")
        logger.info(f"cuDNN Version: {system_info['cudnn_version']}")
        
        if 'gpu_info' in system_info:
            gpu_info = system_info['gpu_info']
            logger.info(f"GPU Count: {gpu_info['device_count']}")
            for i, device in enumerate(gpu_info['devices']):
                logger.info(f"GPU {i}: {device['name']} ({device['total_memory'] / 1024**3:.1f} GB)")
    
    logger.info(f"CPU Count: {system_info['cpu_count']} ({system_info['cpu_count_logical']} logical)")
    logger.info(f"Memory: {system_info['memory_total'] / 1024**3:.1f} GB total, "
               f"{system_info['memory_available'] / 1024**3:.1f} GB available")


def log_model_info(
    model: 'torch.nn.Module',
    logger: Optional[logging.Logger] = None
) -> None:
    """Log model architecture information."""
    if logger is None:
        logger = get_logger()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=== Model Information ===")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Log model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    logger.info(f"Model Size: {model_size_mb:.2f} MB")


def log_experiment_config(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """Log experiment configuration."""
    if logger is None:
        logger = get_logger()
    
    logger.info("=== Experiment Configuration ===")
    
    def log_dict(d: Dict[str, Any], prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)


def create_tensorboard_logger(log_dir: str, experiment_name: str) -> 'torch.utils.tensorboard.SummaryWriter':
    """Create TensorBoard logger."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise NIITrainerError("TensorBoard not available. Install with: pip install tensorboard")
    
    log_path = Path(log_dir) / "tensorboard" / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    return SummaryWriter(log_dir=str(log_path))


def create_wandb_logger(
    project: str,
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[list] = None
) -> Any:
    """Create Weights & Biases logger."""
    try:
        import wandb
    except ImportError:
        raise NIITrainerError("Weights & Biases not available. Install with: pip install wandb")
    
    wandb.init(
        project=project,
        name=experiment_name,
        config=config,
        tags=tags or []
    )
    
    return wandb


class LoggerManager:
    """Comprehensive logger manager for experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.log_dir / f"{experiment_name}.log"
        self.logger = setup_logging(log_file=str(log_file))
        
        # Setup tensorboard
        self.tensorboard_writer = None
        if use_tensorboard:
            try:
                self.tensorboard_writer = create_tensorboard_logger(
                    str(self.log_dir), experiment_name
                )
            except NIITrainerError as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
        
        # Setup wandb
        self.wandb_logger = None
        if use_wandb and wandb_project:
            try:
                self.wandb_logger = create_wandb_logger(
                    wandb_project, experiment_name, {}
                )
            except NIITrainerError as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """Log metrics to all configured loggers."""
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {prefix}{metrics_str}")
        
        # Log to tensorboard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                tag = f"{prefix}{key}" if prefix else key
                self.tensorboard_writer.add_scalar(tag, value, step)
        
        # Log to wandb
        if self.wandb_logger:
            import wandb
            log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
            log_dict['step'] = step
            wandb.log(log_dict)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.logger.info("=== Hyperparameters ===")
        for key, value in hparams.items():
            self.logger.info(f"{key}: {value}")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, {})
        
        if self.wandb_logger:
            import wandb
            wandb.config.update(hparams)
    
    def log_model_graph(self, model: 'torch.nn.Module', input_sample: 'torch.Tensor') -> None:
        """Log model computational graph."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")
        
        if self.wandb_logger:
            try:
                import wandb
                wandb.watch(model)
            except Exception as e:
                self.logger.warning(f"Failed to watch model in W&B: {e}")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_logger:
            import wandb
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()