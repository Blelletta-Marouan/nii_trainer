from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from tqdm import tqdm
import numpy as np
import logging
import os

from ..configs.config import TrainerConfig
from .metrics_manager import MetricsManager
from .gradient_manager import GradientManager
from .visualization_manager import VisualizationManager
from .curriculum_manager import CurriculumManager
from ..losses.losses import CascadedLossWithReward


class ModelTrainer:
    """
    Modular model trainer with support for:
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Comprehensive metric tracking and visualization
    - Multi-stage curriculum learning
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            logger: Optional logger for logging information
        """
        self.config = config
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize loss function
        self.criterion = CascadedLossWithReward(config.loss)
        
        # Setup directories
        self.exp_dir = Path(config.save_dir) / config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component managers
        self._init_managers()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.logger.info(f"Trainer initialized with {type(model).__name__}")
        self.logger.info(f"Using device: {config.training.device}")
        self.logger.info(f"Mixed precision training: {config.training.mixed_precision}")
        self.logger.info(f"Gradient accumulation steps: {config.training.batch_accumulation}")
    
    def _init_managers(self):
        """Initialize component managers."""
        # Create metrics manager
        self.metrics_manager = MetricsManager(
            classes=self.config.data.classes,
            logger=self.logger
        )
        
        # Create gradient manager with gradient accumulation
        self.gradient_manager = GradientManager(
            optimizer=self.optimizer,
            accumulation_steps=self.config.training.batch_accumulation,
            use_mixed_precision=self.config.training.mixed_precision,
            device=self.config.training.device,
            logger=self.logger
        )
        
        # Get the scaler from gradient manager
        self.scaler = self.gradient_manager.scaler
        
        # Create visualization manager
        self.visualization_manager = VisualizationManager(
            classes=self.config.data.classes,
            save_dir=self.exp_dir,
            experiment_name=self.config.experiment_name,
            logger=self.logger
        )
        
        # Curriculum manager will be initialized when needed
        self.curriculum_manager = None
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.exp_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_manager.metrics_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        torch.save(
            checkpoint,
            checkpoint_dir / 'latest_checkpoint.pth'
        )
        
        # Save best model if needed
        if is_best:
            self.logger.info(f"New best model found at epoch {self.current_epoch}")
            torch.save(
                checkpoint,
                checkpoint_dir / 'best_model.pth'
            )
            
        # Save metrics history
        with open(self.exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_manager.metrics_history, f, indent=4)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.training.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'metrics_history' in checkpoint:
            self.metrics_manager.metrics_history = checkpoint['metrics_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_metrics = []
        
        # Determine if we're using curriculum learning with stage freezing
        using_curriculum_freezing = (
            hasattr(self, 'curriculum_manager') and 
            self.curriculum_manager is not None and
            self.curriculum_manager.are_stages_frozen()
        )
        
        # Get active stages for curriculum
        active_stages = None
        if using_curriculum_freezing:
            active_stages, _ = self.curriculum_manager.get_active_and_frozen_stages()
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.config.training.device)
                targets = targets.to(self.config.training.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                with autocast(self.config.training.device, enabled=self.config.training.mixed_precision):
                    # Pass active_stages to model.forward when using curriculum
                    if using_curriculum_freezing:
                        predictions = self.model(images, active_stages=active_stages)
                    else:
                        predictions = self.model(images)
                    
                    # When using curriculum with freezing, only compute metrics for active stages
                    if using_curriculum_freezing:
                        _, frozen_stages = self.curriculum_manager.get_active_and_frozen_stages()
                        metrics_dict = self.criterion(
                            predictions, 
                            targets, 
                            skip_stages=frozen_stages  # Skip calculating loss for frozen stages
                        )
                    else:
                        # Normal training mode - compute metrics for all stages
                        metrics_dict = self.criterion(predictions, targets)
                    
                    loss = metrics_dict['total_loss']  # This includes weighted loss
                    
                # Backward pass with gradient scaling if mixed precision is enabled
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    
                    # If using curriculum with frozen stages, zero out gradients for frozen stages
                    # to completely eliminate any computational overhead
                    if using_curriculum_freezing:
                        _, frozen_stages = self.curriculum_manager.get_active_and_frozen_stages()
                        for stage_idx in frozen_stages:
                            stage = self.model.stages[stage_idx]
                            for param in stage.parameters():
                                param.grad = None
                                
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # If using curriculum with frozen stages, zero out gradients for frozen stages
                    if using_curriculum_freezing:
                        _, frozen_stages = self.curriculum_manager.get_active_and_frozen_stages()
                        for stage_idx in frozen_stages:
                            stage = self.model.stages[stage_idx]
                            for param in stage.parameters():
                                param.grad = None
                                
                    self.optimizer.step()
                    
                # Update metrics
                total_loss += loss.item()
                all_metrics.append(metrics_dict)
                
                # Format metrics for display in progress bar
                display_metrics = {}
                
                # Always show these core metrics
                if 'base_loss' in metrics_dict:
                    display_metrics['base_loss'] = f"{metrics_dict['base_loss'].item():.4f}"
                    
                if 'loss_w_r' in metrics_dict:
                    display_metrics['loss_w_r'] = f"{metrics_dict['loss_w_r'].item():.4f}"
                    
                if 'total_loss' in metrics_dict:
                    display_metrics['total_loss'] = f"{metrics_dict['total_loss'].item():.4f}"
                    
                if 'dice' in metrics_dict:
                    display_metrics['dice'] = f"{metrics_dict['dice'].item():.3f}"
                    
                # Add per-stage metrics based on curriculum mode
                if using_curriculum_freezing:
                    # Only show metrics for active stages when using curriculum
                    for stage_idx in active_stages:
                        stage_key = f'stage{stage_idx+1}'
                        if f'{stage_key}_dice' in metrics_dict:
                            display_metrics[f'S{stage_idx+1}_dice'] = f"{metrics_dict[f'{stage_key}_dice'].item():.3f}"
                else:
                    # Show metrics for all stages when not using curriculum
                    stages_count = len(self.model.stages) if hasattr(self.model, 'stages') else 1
                    for stage_idx in range(stages_count):
                        stage_key = f'stage{stage_idx+1}'
                        if f'{stage_key}_dice' in metrics_dict:
                            display_metrics[f'S{stage_idx+1}_dice'] = f"{metrics_dict[f'{stage_key}_dice'].item():.3f}"
                
                pbar.set_postfix(display_metrics)
                
                # Log every 100 batches
                if batch_idx % 100 == 0:
                    self.logger.debug(
                        f"Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}, Avg Loss: {total_loss / (batch_idx + 1):.4f}"
                    )
                
        # Calculate average metrics
        avg_metrics = {}
        for key in ['total_loss', 'base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']:
            values = []
            for m in all_metrics:
                if key in m and torch.is_tensor(m[key]):
                    val = m[key].item()
                    if np.isfinite(val) and abs(val) < 1e9:
                        values.append(val)
                elif key in m and isinstance(m[key], (int, float)):
                    val = m[key]
                    if np.isfinite(val) and abs(val) < 1e9:
                        values.append(val)
            
            if values:
                avg_metrics[key] = np.mean(values)
        
        # Add per-stage metrics only for active stages when using curriculum
        if hasattr(self.model, 'stages'):
            num_stages = len(self.model.stages)
            frozen_stages = []
            
            if using_curriculum_freezing:
                _, frozen_stages = self.curriculum_manager.get_active_and_frozen_stages()
                
            for stage_idx in range(num_stages):
                # Skip metrics for frozen stages
                if stage_idx in frozen_stages:
                    continue
                    
                stage_prefix = f"stage{stage_idx+1}"
                for metric in ['dice', 'precision', 'recall', 'iou']:
                    metric_key = f"{stage_prefix}_{metric}"
                    values = []
                    for m in all_metrics:
                        if metric_key in m and torch.is_tensor(m[metric_key]):
                            val = m[metric_key].item()
                            if np.isfinite(val) and abs(val) < 1e9:
                                values.append(val)
                    if values:
                        avg_metrics[metric_key] = np.mean(values)
        
        # Set total_loss to be loss_w_r for evaluation since that's what we train with
        if 'loss_w_r' in avg_metrics:
            avg_metrics['loss'] = avg_metrics['loss_w_r']
        else:
            avg_metrics['loss'] = avg_metrics.get('total_loss', 0.0)
        
        # Update metrics history
        self.metrics_manager.update_history('train', avg_metrics)
        
        # Log epoch metrics
        self.logger.info(
            f"Epoch {self.current_epoch} - Training Loss: {avg_metrics['loss']:.4f}"
        )
        if 'base_loss' in avg_metrics:
            self.logger.info(f"Epoch {self.current_epoch} - Training Base Loss: {avg_metrics['base_loss']:.4f}")
            
        for key, value in avg_metrics.items():
            if key not in ['loss', 'base_loss', 'loss_w_r', 'total_loss']:
                self.logger.debug(f"Training {key}: {value:.4f}")
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (metrics dictionary, validation loss)
        """
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        self.logger.info("Starting validation...")
        
        # Determine if we're using curriculum learning with stage freezing
        using_curriculum_freezing = (
            hasattr(self, 'curriculum_manager') and 
            self.curriculum_manager is not None and
            self.curriculum_manager.are_stages_frozen()
        )
        
        frozen_stages = []
        active_stages = None
        if using_curriculum_freezing:
            active_stages, frozen_stages = self.curriculum_manager.get_active_and_frozen_stages()
        
        # Create progress bar with detailed metrics
        val_pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch_idx, (images, targets) in enumerate(val_pbar):
            images = images.to(self.config.training.device)
            targets = targets.to(self.config.training.device)
            
            # Ensure targets have channel dimension
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            
            # Forward pass - pass active_stages to model.forward when using curriculum
            with autocast(self.config.training.device, enabled=self.config.training.mixed_precision):
                if using_curriculum_freezing:
                    predictions = self.model(images, active_stages=active_stages)
                else:
                    predictions = self.model(images)
                
                # When using curriculum with freezing, only compute metrics for active stages
                if using_curriculum_freezing:
                    metrics_dict = self.criterion(
                        predictions, 
                        targets, 
                        skip_stages=frozen_stages  # Skip calculating metrics for frozen stages
                    )
                else:
                    # Normal validation mode - compute metrics for all stages
                    metrics_dict = self.criterion(predictions, targets)
                
                loss = metrics_dict['total_loss']
            
            # Store results
            total_loss += loss.item()
            all_metrics.append(metrics_dict)
            
            # Format display metrics based on curriculum mode
            display_metrics = {}
            
            # Always show these core metrics from the current active stage
            if 'base_loss' in metrics_dict:
                display_metrics['base_loss'] = f"{metrics_dict['base_loss'].item():.4f}"
                
            if 'loss_w_r' in metrics_dict:
                display_metrics['loss_w_r'] = f"{metrics_dict['loss_w_r'].item():.4f}"
                
            if 'total_loss' in metrics_dict:
                display_metrics['total_loss'] = f"{metrics_dict['total_loss'].item():.4f}"
                
            if 'dice' in metrics_dict:
                display_metrics['dice'] = f"{metrics_dict['dice'].item():.3f}"
            
            if 'precision' in metrics_dict:
                display_metrics['precision'] = f"{metrics_dict['precision'].item():.3f}"
                
            if 'recall' in metrics_dict:
                display_metrics['recall'] = f"{metrics_dict['recall'].item():.3f}"
                
            if 'jaccard' in metrics_dict:
                display_metrics['jaccard'] = f"{metrics_dict['jaccard'].item():.3f}"
                
            # Add per-stage metrics based on curriculum mode
            if using_curriculum_freezing:
                # Only show metrics for active stages when using curriculum
                for stage_idx in active_stages:
                    stage_key = f'stage{stage_idx+1}'
                    if f'{stage_key}_dice' in metrics_dict:
                        display_metrics[f'S{stage_idx+1}_dice'] = f"{metrics_dict[f'{stage_key}_dice'].item():.3f}"
            else:
                # Show metrics for all stages when not using curriculum
                stages_count = len(self.model.stages) if hasattr(self.model, 'stages') else 1
                for stage_idx in range(stages_count):
                    stage_key = f'stage{stage_idx+1}'
                    if f'{stage_key}_dice' in metrics_dict:
                        display_metrics[f'S{stage_idx+1}_dice'] = f"{metrics_dict[f'{stage_key}_dice'].item():.3f}"
            
            val_pbar.set_postfix(display_metrics)
        
        # Calculate average metrics across all batches
        avg_metrics = {}
        for key in ['total_loss', 'base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']:
            values = []
            for m in all_metrics:
                if key in m and torch.is_tensor(m[key]):
                    val = m[key].item()
                    if np.isfinite(val) and abs(val) < 1e9:
                        values.append(val)
                elif key in m and isinstance(m[key], (int, float)):
                    val = m[key]
                    if np.isfinite(val) and abs(val) < 1e9:
                        values.append(val)
            
            if values:
                avg_metrics[key] = np.mean(values)
        
        # Add per-stage metrics
        if hasattr(self.model, 'stages'):
            num_stages = len(self.model.stages)
            for stage_idx in range(num_stages):
                # Only include metrics for active stages in curriculum mode
                if not using_curriculum_freezing or stage_idx not in frozen_stages:
                    stage_prefix = f"stage{stage_idx+1}"
                    for metric in ['dice', 'precision', 'recall', 'iou']:
                        metric_key = f"{stage_prefix}_{metric}"
                        values = []
                        for m in all_metrics:
                            if metric_key in m and torch.is_tensor(m[metric_key]):
                                val = m[metric_key].item()
                                if np.isfinite(val) and abs(val) < 1e9:
                                    values.append(val)
                            elif metric_key in m and isinstance(m[metric_key], (int, float)):
                                val = m[metric_key]
                                if np.isfinite(val) and abs(val) < 1e9:
                                    values.append(val)
                        
                        if values:
                            avg_metrics[metric_key] = np.mean(values)
        
        # Update metrics history
        self.metrics_manager.update_history('val', avg_metrics)
        
        return avg_metrics, avg_metrics.get('total_loss', float('inf'))
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train (defaults to config value)
            
        Returns:
            Dictionary of training results
        """
        if epochs is None:
            epochs = self.config.training.epochs
            
        start_epoch = self.current_epoch
        max_patience = self.config.training.patience
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Early stopping patience: {max_patience}")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics, val_loss = self.validate()
            
            # Check if this is the best model so far
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)
            
            # Update learning rate scheduler if provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch summary with metrics in the requested order
            self.visualization_manager.log_epoch_summary(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
                checkpoint_saved=is_best,
                patience_counter=self.patience_counter,
                patience_limit=max_patience,
                learning_rate=self.gradient_manager.get_lr()
            )
            
            # Plot metrics
            if is_best or epoch % 10 == 0:
                self.visualization_manager.plot_metrics(
                    self.metrics_manager.metrics_history,
                    epoch=epoch
                )
            
            # Check early stopping
            if self.patience_counter >= max_patience:
                self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
                
        self.logger.info("Training completed")
        
        return {
            "best_val_loss": self.best_val_loss,
            "epochs_completed": epoch - start_epoch + 1,
            "early_stopped": self.patience_counter >= max_patience,
            "metrics_history": self.metrics_manager.metrics_history
        }
        
    def _optimize_cuda_memory(self):
        """Clear CUDA cache and optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def train_with_curriculum(
        self,
        stage_schedule,
        learning_rates=None,
        stage_freezing=None
    ):
        """
        Train the model using curriculum learning.
        
        Args:
            stage_schedule: List of (stage_idx, num_epochs) tuples
            learning_rates: Optional list of learning rates for each stage
            stage_freezing: Optional list of booleans for freezing previous stages
            
        Returns:
            Dictionary of training results
        """
        # Initialize curriculum manager if not already done
        if self.curriculum_manager is None:
            self.curriculum_manager = CurriculumManager(
                config=self.config.training,
                model=self.model,
                logger=self.logger
            )
            
            # Configure curriculum with the provided parameters
            curriculum_params = {
                'stage_schedule': stage_schedule,
                'learning_rates': learning_rates,
                'stage_freezing': stage_freezing
            }
            
            self.curriculum_manager.configure_curriculum(curriculum_params)
        
        self.logger.info("Starting curriculum training...")
        
        results = {}
        
        # Train according to the curriculum stages
        for stage_idx, (model_stage_idx, stage_epochs) in enumerate(stage_schedule):
            # Clear CUDA cache before each stage
            self._optimize_cuda_memory()
            
            # Update active stages based on current stage
            active_stages = self.curriculum_manager.update_active_stages(stage_idx)
            
            # Get learning rate for this stage
            lr = self.curriculum_manager.get_learning_rate()
            self.gradient_manager.set_lr(lr)
            
            # Apply stage freezing according to curriculum
            self.curriculum_manager.apply_stage_freezing()
            
            # Verify stage freezing status
            for stage_i, stage in enumerate(self.model.stages):
                requires_grad = any(p.requires_grad for p in stage.parameters())
                status = "unfrozen" if requires_grad else "frozen"
                self.logger.info(f"Stage {stage_i+1} is {status}")
            
            stage_name = f"Stage {model_stage_idx+1}"
            self.logger.info(f"=== Starting {stage_name} training for {stage_epochs} epochs ===")
            self.logger.info(f"Learning rate: {lr}")
            
            # Train this stage
            stage_results = self.train(epochs=stage_epochs)
            results[f"stage_{model_stage_idx+1}"] = stage_results
            
            # Clear CUDA cache after stage completion
            self._optimize_cuda_memory()
            
            # Log current GPU memory usage if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                self.logger.info(f"GPU Memory after stage {model_stage_idx+1}:")
                self.logger.info(f"  Allocated: {memory_allocated:.1f} MB")
                self.logger.info(f"  Reserved: {memory_reserved:.1f} MB")
        
        self.logger.info("Curriculum training completed")
        
        return results
    
    @torch.no_grad()
    def evaluate(self, dataloader=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Optional dataloader to evaluate on (defaults to validation dataloader)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if dataloader is None:
            dataloader = self.val_loader
            
        self.model.eval()
        all_metrics = []
        
        self.logger.info("Evaluating model...")
        
        # Use tqdm for progress tracking
        with tqdm(dataloader, desc="Evaluation") as pbar:
            for images, targets in pbar:
                images = images.to(self.config.training.device)
                targets = targets.to(self.config.training.device)
                
                # Forward pass
                with autocast(self.config.training.device, enabled=self.config.training.mixed_precision):
                    predictions = self.model(images)
                    metrics_dict = self.criterion(predictions, targets)
                
                # Store metrics
                all_metrics.append(metrics_dict)
                
                # Update progress bar
                metrics_display = self.metrics_manager.format_metrics_display(
                    metrics_dict,
                    ordered_metrics=['base_loss', 'total_loss', 'dice', 'precision', 'recall', 'iou']
                )
                pbar.set_postfix(metrics_display)
        
        # Calculate average metrics
        ordered_metrics = ['base_loss', 'total_loss', 'dice', 'precision', 'recall', 'iou']
        avg_metrics = self.metrics_manager.get_epoch_averages(
            all_metrics, 
            len(dataloader),
            prioritize_metrics=ordered_metrics
        )
        
        # Log evaluation results
        self.logger.info("Evaluation results:")
        for metric_name in ordered_metrics:
            if metric_name in avg_metrics:
                self.logger.info(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
        
        return avg_metrics