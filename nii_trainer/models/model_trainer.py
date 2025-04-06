"""
Main trainer class that coordinates all training components.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
import logging

from ..configs.config import TrainerConfig
from .metrics_manager import MetricsManager
from .gradient_manager import GradientManager
from .visualization_manager import VisualizationManager
from .curriculum_manager import CurriculumManager
from .checkpoint_manager import CheckpointManager
from .training_loop import TrainingLoop
from .early_stopping import EarlyStoppingHandler
from ..losses.losses import CascadedLossWithReward

class ModelTrainer:
    """
    Coordinates training components and provides a high-level interface for model training.
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize trainer with all components."""
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
        
        # Initialize state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Curriculum learning state
        self.curriculum_enabled = hasattr(config.training, 'curriculum') and config.training.curriculum.enabled
        self.curriculum_params = config.training.curriculum.__dict__ if self.curriculum_enabled else {}
        self.last_stage = -1  # Track stage transitions
        
        # Initialize component managers
        self._init_managers()
        
        # Initialize gradient manager
        self.gradient_manager = GradientManager(
            optimizer=optimizer,
            accumulation_steps=config.training.batch_accumulation,
            use_mixed_precision=config.training.mixed_precision,
            device=config.training.device,
            logger=logger
        )
        
        self.logger.info(f"Trainer initialized with {type(model).__name__}")
        self.logger.info(f"Using device: {config.training.device}")
        self.logger.info(f"Mixed precision training: {config.training.mixed_precision}")
        
    def _init_managers(self):
        """Initialize all component managers."""
        # Early stopping handler
        self.early_stopping = EarlyStoppingHandler(
            patience=self.config.training.early_stopping["patience"],
            logger=self.logger
        )
        
        # Training loop manager
        self.training_loop = TrainingLoop(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.config.training.device,
            use_mixed_precision=self.config.training.mixed_precision,
            logger=self.logger
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            exp_dir=self.exp_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            logger=self.logger,
            config=self.config  # Pass the configuration for architecture validation
        )
        
        # Metrics manager
        self.metrics_manager = MetricsManager(
            classes=self.config.data.classes,
            logger=self.logger
        )
        
        # Visualization manager
        self.visualization_manager = VisualizationManager(
            classes=self.config.data.classes,
            save_dir=self.exp_dir,
            experiment_name=self.config.experiment_name,
            logger=self.logger
        )
        
        # Curriculum manager (initialized when needed)
        self.curriculum_manager = None
        
    def train(self):
        """Train the model."""
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("Starting model training")
            if self.curriculum_enabled:
                self.logger.info("Using curriculum learning with stages:")
                for i, (stage_idx, epochs) in enumerate(self.curriculum_params["stage_schedule"]):
                    self.logger.info(f"Stage {stage_idx + 1}: {epochs} epochs, targeting {self.model.stages[stage_idx].config.target_class}")
            else:
                self.logger.info("Using standard training approach (no curriculum)")
                self.logger.info(f"Training all stages simultaneously for {self.config.training.epochs} epochs")
                for i, stage in enumerate(self.model.stages):
                    self.logger.info(f"Stage {i + 1}: targeting {stage.config.target_class}")
            self.logger.info("=" * 80)

        # Initialize dictionary to store all metrics across epochs for averaging
        all_epochs_metrics = {'train': {}, 'val': {}}
        
        try:
            while not self.should_stop():
                # Get current stage if using curriculum learning
                if self.curriculum_enabled:
                    stage_idx = self.get_current_stage()
                    if stage_idx != self.last_stage:
                        self.last_stage = stage_idx
                        stage_target = self.model.stages[stage_idx].config.target_class
                        self.logger.info("=" * 80)
                        self.logger.info(f"Transitioning to Stage {stage_idx + 1}")
                        self.logger.info(f"Target class: {stage_target}")
                        self.logger.info(f"Learning rate: {self.gradient_manager.get_lr():.2e}")
                        self.logger.info("=" * 80)

                # Train for one epoch using the training loop
                train_metrics = self.training_loop.train_epoch(
                    self.train_loader,
                    self.current_epoch,
                    self.curriculum_manager if self.curriculum_enabled else None
                )
                
                # Validate the epoch
                val_metrics, val_loss = self.training_loop.validate(
                    self.val_loader,
                    self.curriculum_manager if self.curriculum_enabled else None
                )

                # Store metrics for each epoch for later averaging (non-curriculum mode)
                if not self.curriculum_enabled:
                    for metric_name, value in train_metrics.items():
                        if metric_name not in all_epochs_metrics['train']:
                            all_epochs_metrics['train'][metric_name] = []
                        all_epochs_metrics['train'][metric_name].append(value)
                    
                    for metric_name, value in val_metrics.items():
                        if metric_name not in all_epochs_metrics['val']:
                            all_epochs_metrics['val'][metric_name] = []
                        all_epochs_metrics['val'][metric_name].append(value)

                # Log epoch metrics with consistent formatting
                self.logger.info("=" * 80)
                self.logger.info(f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
                if self.curriculum_enabled:
                    stage_idx = self.get_current_stage()
                    self.logger.info(f"Current Stage: {stage_idx + 1} ({self.model.stages[stage_idx].config.target_class})")

                # Display core metrics with consistent 4-decimal formatting
                self.logger.info("Training Metrics:")
                for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                    if metric in train_metrics:
                        self.logger.info(f"  {metric}: {train_metrics[metric]:.4f}")
                
                self.logger.info("Validation Metrics:")
                for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                    if metric in val_metrics:
                        self.logger.info(f"  {metric}: {val_metrics[metric]:.4f}")

                # Per-stage metrics if available
                if any(key.startswith('stage') for key in train_metrics):
                    self.logger.info("Per-Stage Metrics:")
                    for i in range(len(self.model.stages)):
                        stage_prefix = f"stage{i+1}_"
                        stage_metrics = {k.replace(stage_prefix, ''): v 
                                      for k, v in train_metrics.items() 
                                      if k.startswith(stage_prefix)}
                        if stage_metrics:
                            self.logger.info(f"  Stage {i+1}:")
                            for metric, value in stage_metrics.items():
                                self.logger.info(f"    {metric}: {value:.4f}")

                self.logger.info("=" * 80)

                # Update learning rate if using scheduler
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if self.early_stopping(val_loss, self.current_epoch):
                    self.logger.info("Early stopping triggered")
                    break

                # Update best model if needed
                current_val_loss = val_loss
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=self.current_epoch,
                        best_val_loss=self.best_val_loss,
                        metrics_history=self.metrics_manager.metrics_history,
                        is_best=True
                    )
                    self.logger.info(f"New best model saved! (val_loss: {current_val_loss:.4f})")

                self.current_epoch += 1

                # Update metrics history
                self.metrics_manager.update_history('train', train_metrics)
                self.metrics_manager.update_history('val', val_metrics)

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

        # After training completes, display average metrics if not using curriculum
        if not self.curriculum_enabled and self.current_epoch > 0:
            self.logger.info("=" * 80)
            self.logger.info("TRAINING SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total epochs completed: {self.current_epoch}")
            
            # Calculate and display average metrics across all epochs
            self.logger.info("Average Training Metrics:")
            for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                if metric in all_epochs_metrics['train'] and all_epochs_metrics['train'][metric]:
                    avg_value = sum(all_epochs_metrics['train'][metric]) / len(all_epochs_metrics['train'][metric])
                    self.logger.info(f"  {metric}: {avg_value:.4f}")
            
            self.logger.info("Average Validation Metrics:")
            for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                if metric in all_epochs_metrics['val'] and all_epochs_metrics['val'][metric]:
                    avg_value = sum(all_epochs_metrics['val'][metric]) / len(all_epochs_metrics['val'][metric])
                    self.logger.info(f"  {metric}: {avg_value:.4f}")
                    
            # Calculate and display per-stage average metrics 
            stage_prefixes = [f"stage{i+1}_" for i in range(len(self.model.stages))]
            
            # Check if we have per-stage metrics
            has_stage_metrics = False
            for prefix in stage_prefixes:
                for phase in ['train', 'val']:
                    if any(k.startswith(prefix) for k in all_epochs_metrics[phase]):
                        has_stage_metrics = True
                        break
                if has_stage_metrics:
                    break
                    
            if has_stage_metrics:
                self.logger.info("Average Per-Stage Metrics:")
                
                for i, prefix in enumerate(stage_prefixes):
                    # Only process stages that have data
                    stage_has_data = False
                    for phase in ['train', 'val']:
                        if any(k.startswith(prefix) for k in all_epochs_metrics[phase]):
                            stage_has_data = True
                            break
                            
                    if not stage_has_data:
                        continue
                
                    self.logger.info(f"  Stage {i+1} ({self.model.stages[i].config.target_class}):")
                    
                    # Training metrics for this stage
                    self.logger.info("    Training:")
                    for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                        metric_key = f"{prefix}{metric}"
                        if metric_key in all_epochs_metrics['train'] and all_epochs_metrics['train'][metric_key]:
                            avg_value = sum(all_epochs_metrics['train'][metric_key]) / len(all_epochs_metrics['train'][metric_key])
                            self.logger.info(f"      {metric}: {avg_value:.4f}")
                    
                    # Validation metrics for this stage
                    self.logger.info("    Validation:")
                    for metric in ['loss', 'dice', 'precision', 'recall', 'iou']:
                        metric_key = f"{prefix}{metric}"
                        if metric_key in all_epochs_metrics['val'] and all_epochs_metrics['val'][metric_key]:
                            avg_value = sum(all_epochs_metrics['val'][metric_key]) / len(all_epochs_metrics['val'][metric_key])
                            self.logger.info(f"      {metric}: {avg_value:.4f}")
                            
            self.logger.info("=" * 80)
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info("=" * 80)

        return self.metrics_manager.metrics_history
        
    def train_with_curriculum(
        self, 
        stage_schedule: List[Tuple[int, int]], 
        learning_rates: Optional[List[float]] = None,
        stage_freezing: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """
        Train the model using curriculum learning approach.
        
        Args:
            stage_schedule: List of (stage_idx, num_epochs) tuples defining the training schedule
            learning_rates: Optional list of learning rates for each stage
            stage_freezing: Optional list of booleans indicating whether to freeze previous stages
            
        Returns:
            Dictionary of training results
        """
        if self.curriculum_manager is None:
            self.logger.warning("Curriculum manager not initialized. Creating one now.")
            from .curriculum_manager import CurriculumManager
            self.curriculum_manager = CurriculumManager(
                config=self.config.training,
                model=self.model,
                logger=self.logger
            )
            
        # Configure curriculum with provided parameters
        curriculum_params = {
            'stage_schedule': stage_schedule
        }
        if learning_rates:
            curriculum_params['learning_rates'] = learning_rates
        if stage_freezing:
            curriculum_params['stage_freezing'] = stage_freezing
            
        self.curriculum_manager.configure_curriculum(curriculum_params)
        
        # Train through each curriculum stage
        self.logger.info(f"Starting curriculum training with {len(stage_schedule)} stages")
        
        results = {}
        total_epochs = sum(epochs for _, epochs in stage_schedule)
        epoch_offset = 0
        
        for curriculum_stage, (stage_idx, num_epochs) in enumerate(stage_schedule):
            self.logger.info(f"Curriculum stage {curriculum_stage+1}/{len(stage_schedule)}: "
                            f"Training stage {stage_idx+1} for {num_epochs} epochs")
            
            # Update active stages for this curriculum stage
            self.curriculum_manager.update_active_stages(curriculum_stage)
            
            # Always apply stage freezing in curriculum learning for GPU efficiency
            # This ensures only the current stage is active and unfrozen
            self.curriculum_manager.apply_stage_freezing()
            
            # Get current learning rate for this stage
            if learning_rates and curriculum_stage < len(learning_rates):
                lr = learning_rates[curriculum_stage]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.logger.info(f"Setting learning rate to {lr}")
            
            # Log model parameter status for better transparency
            self._log_model_parameter_status()
                
            # Train for this curriculum stage
            stage_results = {}
            
            for epoch in range(num_epochs):
                self.current_epoch = epoch_offset + epoch
                
                # Training phase
                train_metrics = self.training_loop.train_epoch(
                    self.train_loader, 
                    self.current_epoch,
                    self.curriculum_manager
                )
                
                # Validation phase
                val_metrics, val_loss = self.training_loop.validate(
                    self.val_loader,
                    self.curriculum_manager
                )
                
                # Update learning rate if using scheduler
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                # Check for early stopping and best model
                should_stop = self.early_stopping(val_loss, self.current_epoch)
                is_best = self.early_stopping.should_save_checkpoint
                
                if is_best:
                    self.best_val_loss = val_loss
                    
                # Update metrics history
                self.metrics_manager.update_history('train', train_metrics)
                self.metrics_manager.update_history('val', val_metrics)
                
                # Save checkpoint
                self.checkpoint_manager.save(
                    current_epoch=self.current_epoch,
                    metrics_history=self.metrics_manager.metrics_history,
                    best_val_loss=self.best_val_loss,
                    is_best=is_best,
                    curriculum_stage=curriculum_stage
                )
                
                # Log epoch summary
                self.visualization_manager.log_epoch_summary(
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch=self.current_epoch,
                    checkpoint_saved=is_best,
                    patience_counter=self.early_stopping.counter,
                    patience_limit=self.config.training.early_stopping["patience"],
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    curriculum_stage=curriculum_stage,
                    stage_idx=stage_idx
                )
                
                # Check if training should stop
                if should_stop:
                    self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                    break
                    
                # Store results for this epoch
                stage_results[f"epoch_{epoch}"] = {
                    "train": train_metrics,
                    "val": val_metrics
                }
            
            # Update epoch offset for next stage
            epoch_offset += num_epochs
            
            # Reset early stopping counter between stages
            self.early_stopping.reset()
            
            # Store results for this curriculum stage
            results[f"curriculum_stage_{curriculum_stage}"] = {
                "stage_idx": stage_idx,
                "epochs": stage_results
            }
            
            self.logger.info(f"Completed curriculum stage {curriculum_stage+1} (model stage {stage_idx+1})")
            
        self.logger.info("Curriculum training completed")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return results
        
    def _log_model_parameter_status(self):
        """Log the trainable/frozen status of model parameters for better transparency."""
        if not hasattr(self.model, 'stages'):
            self.logger.info("Model doesn't have multiple stages, skipping parameter status logging")
            return
            
        # Count trainable parameters per stage
        total_params = 0
        trainable_params = 0
        stage_params = []
        stage_trainable = []
        
        for stage_idx, stage in enumerate(self.model.stages):
            stage_total = sum(p.numel() for p in stage.parameters())
            stage_train = sum(p.numel() for p in stage.parameters() if p.requires_grad)
            
            stage_params.append(stage_total)
            stage_trainable.append(stage_train)
            
            total_params += stage_total
            trainable_params += stage_train
            
            status = "TRAINABLE" if stage_train == stage_total else "FROZEN"
            pct = (stage_train / stage_total * 100) if stage_total > 0 else 0
            
            self.logger.info(f"Stage {stage_idx+1}: {status} - {stage_train:,}/{stage_total:,} parameters ({pct:.1f}%)")
        
        frozen_pct = 100 - (trainable_params / total_params * 100) if total_params > 0 else 0
        self.logger.info(f"Total trainable parameters: {trainable_params:,}/{total_params:,} ({100-frozen_pct:.1f}%)")
        self.logger.info(f"Total frozen parameters: {total_params-trainable_params:,}/{total_params:,} ({frozen_pct:.1f}%)")
        
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """Load a checkpoint and restore training state."""
        checkpoint = self.checkpoint_manager.init_from_checkpoint(checkpoint_path)
        
        if checkpoint:
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            if 'metrics_history' in checkpoint:
                self.metrics_manager.metrics_history = checkpoint['metrics_history']
                
    def configure_curriculum(self, params: Dict[str, Any]) -> None:
        """Configure curriculum learning if needed."""
        if params:
            if self.curriculum_manager is None:
                self.curriculum_manager = CurriculumManager(
                    config=self.config.training,
                    model=self.model,
                    logger=self.logger
                )
            self.curriculum_manager.configure_curriculum(params)
            
    def get_current_stage(self) -> int:
        """Get the current active stage in curriculum learning."""
        if not self.curriculum_enabled:
            return 0
            
        total_epochs = 0
        for stage_idx, num_epochs in self.curriculum_params["stage_schedule"]:
            total_epochs += num_epochs
            if self.current_epoch < total_epochs:
                return stage_idx
        return len(self.model.stages) - 1  # Return last stage if beyond schedule
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.curriculum_enabled:
            # Get total epochs from curriculum schedule
            total_epochs = sum(epochs for _, epochs in self.curriculum_params["stage_schedule"])
            return self.current_epoch >= total_epochs
        else:
            return self.current_epoch >= self.config.training.epochs

    @property
    def max_epochs(self) -> int:
        """Get the total number of epochs for training."""
        if self.curriculum_enabled:
            return sum(epochs for _, epochs in self.curriculum_params["stage_schedule"])
        return self.config.training.epochs