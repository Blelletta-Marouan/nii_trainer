import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json

class CheckpointManager:
    """Manages model checkpoints including saving, loading, and cleanup."""
    
    def __init__(
        self,
        exp_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[logging.Logger] = None,
        max_checkpoints: int = 5,
        config: Optional[Any] = None
    ):
        self.exp_dir = exp_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger(__name__)
        self.max_checkpoints = max_checkpoints
        self.config = config  # Store the configuration for architecture validation
        
        self.checkpoint_dir = exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, current_epoch: int, metrics_history: Dict, best_val_loss: float, is_best: bool = False, curriculum_stage: Optional[int] = None):
        """
        Save model checkpoint with custom timing and cleanup.
        
        Args:
            current_epoch: Current training epoch
            metrics_history: Dictionary of training metrics history
            best_val_loss: Best validation loss encountered
            is_best: Whether this is the best model so far
            curriculum_stage: Optional curriculum stage being trained
        """
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics_history': metrics_history
        }
        
        # Add configuration information for architecture validation
        if self.config is not None:
            checkpoint['config'] = self.config
        
        # Add curriculum stage information if provided
        if curriculum_stage is not None:
            checkpoint['curriculum_stage'] = curriculum_stage
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if current_epoch % 5 == 0:  # Save every 5 epochs
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{current_epoch}.pth'
            torch.save(checkpoint, periodic_path)
        
        if is_best:
            self.logger.info(f"New best model found at epoch {current_epoch}")
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
        # Save metrics history separately for easier access
        metrics_path = self.exp_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)
            
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and return the checkpoint data."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if it exists
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.logger.info(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            self.logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise
            
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(str(x)), reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                checkpoint.unlink()
            except Exception as e:
                self.logger.warning(f"Could not remove old checkpoint {checkpoint}: {str(e)}")
                
    def init_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Initialize training from a checkpoint if it exists."""
        if checkpoint_path is None:
            # Look for latest checkpoint
            latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
            if not latest_path.exists():
                return {}
            checkpoint_path = str(latest_path)
            
        return self.load(checkpoint_path)