"""Factory functions for creating and configuring trainers."""
import logging
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader

from .model_trainer import ModelTrainer
from .model_utils import create_model, initialize_model_optimizer
from .checkpoint_manager import CheckpointManager
from .training_loop import TrainingLoop
from .early_stopping import EarlyStoppingHandler
from .metrics_manager import MetricsManager
from .visualization_manager import VisualizationManager
from .curriculum_manager import CurriculumManager
from ..configs.config import TrainerConfig

def create_trainer(
    config: TrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Create and configure a complete model trainer with all components.
    
    Args:
        config: Complete trainer configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        logger: Optional logger for logging information
        
    Returns:
        Configured ModelTrainer instance
    """
    if logger:
        logger.info("Creating model trainer...")
        
    # Create model and optimizer
    model = create_model(config.cascade, config.training.device, logger)
    optimizer = initialize_model_optimizer(model, config, logger)
    
    # Create scheduler if configured
    scheduler = None
    if config.training.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=config.training.reduce_lr_patience,
            verbose=True
        )
    
    # Initialize trainer with base components
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )
    
    # Add optional curriculum learning
    if config.training.curriculum.enabled:
        if logger:
            logger.info("Configuring curriculum learning...")
            
        curriculum_params = {
            'stage_schedule': config.training.curriculum.stage_schedule,
            'learning_rates': config.training.curriculum.learning_rates,
            'stage_freezing': config.training.curriculum.stage_freezing
        }
        trainer.configure_curriculum(curriculum_params)
    
    return trainer

def load_trainer_from_checkpoint(
    config: TrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: str,
    logger: Optional[logging.Logger] = None
) -> ModelTrainer:
    """
    Create a trainer and restore its state from a checkpoint.
    
    Args:
        config: Trainer configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        checkpoint_path: Path to checkpoint file
        logger: Optional logger
        
    Returns:
        Restored ModelTrainer instance
    """
    # Create new trainer
    trainer = create_trainer(config, train_loader, val_loader, logger)
    
    # Load checkpoint state
    trainer.load_checkpoint(checkpoint_path)
    
    return trainer