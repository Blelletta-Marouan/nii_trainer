from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import torch
from torch.utils.data import DataLoader

from ..configs.config import TrainerConfig
from ..models.trainer import ModelTrainer
from ..models.model_utils import save_checkpoint

def setup_trainer(
    config: TrainerConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    start_epoch: int = 0,
    best_val_loss: float = float('inf'),
    metrics_history: Optional[Dict[str, List]] = None,
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Set up a trainer for model training.
    
    Args:
        config: Trainer configuration
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        start_epoch: Starting epoch (for resuming training)
        best_val_loss: Best validation loss (for resuming training)
        metrics_history: Metrics history (for resuming training)
        logger: Logger for logging information
        
    Returns:
        Configured trainer instance
    """
    if logger:
        logger.info("Setting up trainer...")
    
    # Create scheduler if enabled in config
    scheduler = None
    if hasattr(config.training, 'use_scheduler') and config.training.use_scheduler:
        if logger:
            logger.info("Creating learning rate scheduler...")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5 if not hasattr(config.training, 'scheduler_patience') else config.training.scheduler_patience,
            verbose=True
        )
    
    # Create trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )
    
    # Set trainer state if resuming training
    if start_epoch > 0:
        trainer.current_epoch = start_epoch
        trainer.best_val_loss = best_val_loss
        
        if metrics_history:
            trainer.metrics_history = metrics_history
            
        if logger:
            logger.info(f"Resuming training from epoch {start_epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return trainer

def train_model(
    trainer: ModelTrainer,
    config: TrainerConfig,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Train model using standard training approach.
    
    Args:
        trainer: Configured trainer instance
        config: Trainer configuration
        checkpoint_dir: Directory to save checkpoints
        logger: Logger for logging information
        
    Returns:
        Trained trainer instance
    """
    if logger:
        logger.info("Starting model training...")
        logger.info(f"Training for {config.training.epochs} epochs")
        logger.info(f"Early stopping patience: {config.training.patience}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Train model
    trainer.trainer()
    
    if logger:
        logger.info("Training completed")
    
    return trainer

def train_with_curriculum(
    trainer: ModelTrainer,
    stage_schedule: List[Tuple[int, int]],
    learning_rates: Optional[List[float]] = None,
    stage_freezing: Optional[List[bool]] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Train model using curriculum learning approach.
    
    Args:
        trainer: Configured trainer instance
        stage_schedule: List of (stage_idx, num_epochs) tuples
        learning_rates: List of learning rates for each stage
        stage_freezing: Whether to freeze previous stages
        checkpoint_dir: Directory to save checkpoints
        logger: Logger for logging information
        
    Returns:
        Trained trainer instance
    """
    if logger:
        logger.info("Starting curriculum training...")
        
        for i, (stage_idx, num_epochs) in enumerate(stage_schedule):
            logger.info(f"Stage {i+1}: Training stage {stage_idx+1} for {num_epochs} epochs")
            if learning_rates and i < len(learning_rates):
                logger.info(f"  Learning rate: {learning_rates[i]}")
            if stage_freezing and i < len(stage_freezing):
                logger.info(f"  Freeze previous stages: {stage_freezing[i]}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Train with curriculum
    trainer.trainer(
        stage_schedule=stage_schedule,
        learning_rates=learning_rates,
        stage_freezing=stage_freezing
    )
    
    if logger:
        logger.info("Curriculum training completed")
    
    return trainer

def evaluate_model(
    trainer: ModelTrainer,
    config: TrainerConfig,
    test_loader: Optional[DataLoader] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Evaluate trained model on test data.
    
    Args:
        trainer: Trained trainer instance
        config: Trainer configuration
        test_loader: Test data loader (if None, uses validation loader)
        logger: Logger for logging information
        
    Returns:
        Dictionary of evaluation metrics
    """
    if logger:
        logger.info("Evaluating model...")
    
    loader = test_loader if test_loader is not None else trainer.val_loader
    
    # Set model to evaluation mode
    trainer.model.eval()
    
    # Evaluate model
    with torch.no_grad():
        metrics = trainer.validate()
    
    # Log metrics
    if logger:
        logger.info("Evaluation results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics