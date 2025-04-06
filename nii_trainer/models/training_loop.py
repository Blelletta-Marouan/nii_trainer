import torch
import torch.nn as nn
from torch.amp import autocast
import logging
from typing import Any, Dict, Optional, List, Tuple
from tqdm import tqdm
import numpy as np

from nii_trainer.models.metrics_manager import MetricAggregator

class TrainingLoop:
    """Handles training iterations, metric tracking, and gradient updates."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        use_mixed_precision: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize mixed precision scaler if needed
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        current_epoch: int,
        curriculum_manager: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            current_epoch: Current epoch number
            curriculum_manager: Optional curriculum manager for staged training
        
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Track metrics
        batch_metrics = []
        
        # Get active stages if using curriculum
        active_stages = None
        skip_stages = None
        if curriculum_manager is not None:
            active_stages, frozen_stages = curriculum_manager.get_active_and_frozen_stages()
            # All stages that aren't active should be skipped
            skip_stages = frozen_stages
        
        # Initialize the metrics aggregator
        metrics_aggregator = MetricAggregator()
        
        # Determine if we're using curriculum or not to control which metrics to display
        using_curriculum = curriculum_manager is not None and curriculum_manager.are_stages_frozen()
        
        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {current_epoch}")
        
        # Loop through batches
        for batch_idx, batch in enumerate(pbar):
            # Get data and move to device
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                outputs = self.model(inputs)
                
                # Compute loss with skip_stages parameter
                metrics = self.criterion(outputs, targets, skip_stages=skip_stages)
                loss = metrics["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            
            # Mixed precision backward
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            metrics_aggregator.update(metrics)
            batch_metrics.append(metrics)
            
            # Prepare metrics for display based on mode (curriculum vs non-curriculum)
            display_metrics = {}
            
            if using_curriculum:
                # For curriculum learning, show the overall metrics
                core_metrics = ['base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']
                for metric_name in core_metrics:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if torch.is_tensor(value):
                            value = value.item()
                        # Format to 4 decimal places
                        display_metrics[metric_name] = f"{value:.4f}"
            else:
                # For non-curriculum learning, ONLY show the per-stage metrics
                # Look for metrics with stage prefix (stage1_, stage2_, etc.)
                for key in metrics:
                    if key.startswith('stage') and '_' in key:
                        # Only display core metrics in progress bar to avoid clutter
                        metric_name = key.split('_', 1)[1]  # Extract name part after stage prefix
                        if metric_name in ['dice', 'precision', 'recall', 'jaccard']:
                            value = metrics[key]
                            if torch.is_tensor(value):
                                value = value.item()
                            # Add to display with abbreviated stage identifier (s1, s2...)
                            stage_num = key.split('_')[0].replace('stage', '')
                            display_metrics[f"s{stage_num}_{metric_name}"] = f"{value:.4f}"
                
                # Also show loss metrics since they're important
                if 'base_loss' in metrics:
                    value = metrics['base_loss']
                    if torch.is_tensor(value):
                        value = value.item()
                    display_metrics['base_loss'] = f"{value:.4f}"
                
                if 'loss_w_r' in metrics:
                    value = metrics['loss_w_r']
                    if torch.is_tensor(value):
                        value = value.item()
                    display_metrics['loss_w_r'] = f"{value:.4f}"
            
            # Update the progress bar
            pbar.set_postfix(display_metrics)
        
        # Compute average metrics for the entire epoch
        avg_metrics = metrics_aggregator.get_averages()
        
        # Print the current active stage information if using curriculum
        if curriculum_manager is not None and hasattr(curriculum_manager, 'current_primary_stage'):
            primary_stage = curriculum_manager.current_primary_stage
            self.logger.info(f"Training active stage: {primary_stage+1}")
        
        return avg_metrics
        
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        curriculum_manager = None
    ) -> Tuple[Dict[str, float], float]:
        """Run validation."""
        self.model.eval()
        
        # Initialize metrics aggregator
        metrics_aggregator = MetricAggregator()
        
        # Determine curriculum settings
        using_curriculum = (
            curriculum_manager is not None and 
            curriculum_manager.are_stages_frozen()
        )
        
        skip_stages = []
        if using_curriculum:
            _, frozen_stages = curriculum_manager.get_active_and_frozen_stages()
            skip_stages = frozen_stages
        
        val_pbar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (images, targets) in enumerate(val_pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                outputs = self.model(images)
                metrics = self.criterion(outputs, targets, skip_stages=skip_stages)
            
            # Update metrics aggregator
            metrics_aggregator.update(metrics)
            
            # Prepare metrics for display based on mode (curriculum vs non-curriculum)
            display_metrics = {}
            
            if using_curriculum:
                # For curriculum learning, show the overall metrics
                core_metrics = ['base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']
                for metric_name in core_metrics:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if torch.is_tensor(value):
                            value = value.item()
                        # Format to 4 decimal places
                        display_metrics[metric_name] = f"{value:.4f}"
            else:
                # For non-curriculum learning, ONLY show the per-stage metrics
                # Look for metrics with stage prefix (stage1_, stage2_, etc.)
                for key in metrics:
                    if key.startswith('stage') and '_' in key:
                        # Only display core metrics in progress bar to avoid clutter
                        metric_name = key.split('_', 1)[1]  # Extract name part after stage prefix
                        if metric_name in ['dice', 'precision', 'recall', 'jaccard']:
                            value = metrics[key]
                            if torch.is_tensor(value):
                                value = value.item()
                            # Add to display with abbreviated stage identifier (s1, s2...)
                            stage_num = key.split('_')[0].replace('stage', '')
                            display_metrics[f"s{stage_num}_{metric_name}"] = f"{value:.4f}"
                
                # Also show loss metrics since they're important
                if 'base_loss' in metrics:
                    value = metrics['base_loss']
                    if torch.is_tensor(value):
                        value = value.item()
                    display_metrics['base_loss'] = f"{value:.4f}"
                
                if 'loss_w_r' in metrics:
                    value = metrics['loss_w_r']
                    if torch.is_tensor(value):
                        value = value.item()
                    display_metrics['loss_w_r'] = f"{value:.4f}"
            
            # Update progress bar
            val_pbar.set_postfix(display_metrics)
            
        # Compute average metrics
        avg_metrics = metrics_aggregator.get_averages()
        
        return avg_metrics, avg_metrics.get('total_loss', float('inf'))
        
    def _train_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        active_stages: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Process a single training batch."""
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        with autocast(self.device, enabled=self.use_mixed_precision):
            predictions = self.model(images, active_stages=active_stages)
            metrics_dict = self.criterion(predictions, targets)
            loss = metrics_dict['total_loss']
        
        # Backward pass with gradient scaling if enabled
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            
        return metrics_dict
        
    def _validate_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Process a single validation batch."""
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Ensure targets have channel dimension
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Forward pass with mixed precision
        with autocast(self.device, enabled=self.use_mixed_precision):
            predictions = self.model(images)
            metrics_dict = self.criterion(predictions, targets)
            
        return metrics_dict
        
    def _init_running_metrics(self) -> Dict[str, float]:
        """Initialize running metrics dictionary."""
        return {
            'base_loss': 0.0,
            'loss_w_r': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'jaccard': 0.0
        }
        
    def _update_running_metrics(
        self,
        running_metrics: Dict[str, float],
        batch_metrics: Dict[str, float],
        batch_idx: int
    ) -> None:
        """Update running averages with new batch metrics."""
        current_batch = batch_idx + 1
        for metric_name in running_metrics:
            if metric_name in batch_metrics:
                value = batch_metrics[metric_name]
                if torch.is_tensor(value):
                    value = value.item()
                running_metrics[metric_name] = (
                    (running_metrics[metric_name] * batch_idx + value) / current_batch
                )
                
    def _update_progress_bar(
        self,
        pbar: tqdm,
        running_metrics: Dict[str, float]
    ) -> None:
        """Update progress bar with current metrics."""
        display_metrics = {}
        for metric_name, value in running_metrics.items():
            if metric_name in ['base_loss', 'loss_w_r']:
                display_metrics[metric_name] = f"{value:.4f}"
            else:
                display_metrics[metric_name] = f"{value:.3f}"
        pbar.set_postfix(display_metrics)
        
    def _log_batch_metrics(
        self,
        batch_idx: int,
        total_batches: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log metrics for the current batch."""
        self.logger.debug(
            f"Batch {batch_idx}/{total_batches}, "
            f"Loss: {metrics['total_loss'].item():.4f}"
        )
        
    def _compute_epoch_metrics(
        self,
        running_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute final metrics for the epoch."""
        # Filter out invalid values and compute averages
        avg_metrics = {}
        for key, value in running_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                avg_metrics[key] = float(value)
                
        # Ensure total_loss is set for model evaluation
        if 'loss_w_r' in avg_metrics:
            avg_metrics['total_loss'] = avg_metrics['loss_w_r']
            
        return avg_metrics