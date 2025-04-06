"""Training utility functions for model training and evaluation."""
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader

from ..configs.config import TrainerConfig
from .model_trainer import ModelTrainer

def setup_trainer(
    config: TrainerConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
        logger: Logger for logging information
        
    Returns:
        Configured trainer instance
    """
    if logger:
        logger.info("Setting up trainer...")
    
    # Create scheduler if enabled in config
    scheduler = None
    if getattr(config.training, 'scheduler_type', None) == 'plateau':
        if logger:
            logger.info("Creating learning rate scheduler...")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=config.training.reduce_lr_patience,
            verbose=True
        )
    
    # Create trainer instance
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )
    
    return trainer

def train_model(
    trainer: ModelTrainer,
    checkpoint_path: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Train model using standard training approach.
    
    Args:
        trainer: Configured trainer instance
        checkpoint_path: Optional path to checkpoint to load
        logger: Logger for logging information
        
    Returns:
        Trained trainer instance
    """
    if logger:
        logger.info("Starting model training...")
    
    # Load checkpoint if provided or available
    trainer.load_checkpoint(checkpoint_path)
    
    # Train model
    trainer.train()
    
    if logger:
        logger.info("Training completed")
    
    return trainer

def train_with_curriculum(
    trainer: ModelTrainer,
    curriculum_params: Dict[str, Any],
    checkpoint_path: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Train model using curriculum learning approach.
    
    Args:
        trainer: Configured trainer instance
        curriculum_params: Curriculum learning parameters
        checkpoint_path: Optional path to checkpoint to load
        logger: Logger for logging information
        
    Returns:
        Trained trainer instance
    """
    if logger:
        logger.info("Starting curriculum training...")
    
    # Configure curriculum learning
    trainer.configure_curriculum(curriculum_params)
    
    # Load checkpoint if provided
    trainer.load_checkpoint(checkpoint_path)
    
    # Train model
    trainer.train()
    
    if logger:
        logger.info("Curriculum training completed")
    
    return trainer

def evaluate_model(
    trainer: ModelTrainer,
    test_loader: Optional[DataLoader] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Evaluate trained model.
    
    Args:
        trainer: Trained trainer instance
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
    
    # Evaluate model using training loop
    metrics, _ = trainer.training_loop.validate(loader)
    
    # Log metrics
    if logger:
        logger.info("Evaluation results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics