"""
Base trainer classes for NII-Trainer.

This module provides abstract base classes and common functionality
for all training implementations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from pathlib import Path

from ..core.exceptions import TrainingError
from ..core.registry import register_trainer
from ..core.config import TrainingConfig, GlobalConfig


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(self, config: GlobalConfig, model: nn.Module, **kwargs):
        self.config = config
        self.model = model
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.training_history = []
        
        # Setup components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training."""
        logger = logging.getLogger(f"nii_trainer.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        pass
    
    @abstractmethod
    def setup_criterion(self) -> None:
        """Setup loss function."""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        pass
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Setup training components
        self.setup_optimizer()
        self.setup_criterion()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Mixed precision setup
        if self.config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        try:
            for epoch in range(self.config.training.max_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch(train_loader, epoch)
                self.train_metrics = train_metrics
                
                # Validate epoch
                if val_loader is not None:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                    self.val_metrics = val_metrics
                else:
                    val_metrics = {}
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        monitor_metric = val_metrics.get(
                            self.config.training.early_stopping_metric, 
                            train_metrics.get('loss', 0)
                        )
                        self.scheduler.step(monitor_metric)
                    else:
                        self.scheduler.step()
                
                # Log epoch results
                self._log_epoch_results(epoch, train_metrics, val_metrics)
                
                # Save checkpoint
                is_best = self._check_best_metric(val_metrics if val_metrics else train_metrics)
                self._save_checkpoint(epoch, is_best)
                
                # Early stopping check
                if self._should_early_stop(val_metrics if val_metrics else train_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            self.logger.info("Training completed successfully!")
            return self._get_training_summary()
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]) -> None:
        """Log results for one epoch."""
        log_msg = f"Epoch {epoch:3d}/{self.config.training.max_epochs}"
        
        # Training metrics
        for key, value in train_metrics.items():
            log_msg += f" | train_{key}: {value:.4f}"
        
        # Validation metrics
        for key, value in val_metrics.items():
            log_msg += f" | val_{key}: {value:.4f}"
        
        # Learning rate
        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            log_msg += f" | lr: {lr:.6f}"
        
        self.logger.info(log_msg)
        
        # Store in history
        epoch_data = {
            'epoch': epoch,
            'train_metrics': train_metrics.copy(),
            'val_metrics': val_metrics.copy()
        }
        self.training_history.append(epoch_data)
    
    def _check_best_metric(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far."""
        metric_name = self.config.training.early_stopping_metric
        if metric_name not in metrics:
            return False
        
        current_metric = metrics[metric_name]
        mode = self.config.training.early_stopping_mode
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return True
        
        if mode == "min" and current_metric < self.best_metric:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            return True
        elif mode == "max" and current_metric > self.best_metric:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            return True
        else:
            self.early_stopping_counter += 1
            return False
    
    def _should_early_stop(self, metrics: Dict[str, float]) -> bool:
        """Check if training should be stopped early."""
        patience = self.config.training.early_stopping_patience
        return self.early_stopping_counter >= patience
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with {self.config.training.early_stopping_metric}: {self.best_metric:.4f}")
        
        # Save last checkpoint
        last_path = checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        return {
            'total_epochs': self.current_epoch + 1,
            'best_metric': self.best_metric,
            'final_train_metrics': self.train_metrics,
            'final_val_metrics': self.val_metrics,
            'training_history': self.training_history
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric')
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class BaseMetric:
    """Base class for training metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset metric state."""
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1) -> None:
        """Update metric with new value."""
        self.total += value * n
        self.count += n
    
    def compute(self) -> float:
        """Compute final metric value."""
        if self.count == 0:
            return 0.0
        return self.total / self.count


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise TrainingError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    scheduler_name = config.scheduler.lower()
    
    if scheduler_name == "none" or scheduler_name is None:
        return None
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_patience, gamma=config.scheduler_factor
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=config.early_stopping_mode, 
            patience=config.scheduler_patience, factor=config.scheduler_factor
        )
    elif scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler_factor
        )
    else:
        raise TrainingError(f"Unknown scheduler: {scheduler_name}")