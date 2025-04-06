"""Model creation and initialization utilities."""
import logging
import torch
from typing import Optional
from pathlib import Path
from ..configs.config import CascadeConfig
from ..models.cascaded_unet import FlexibleCascadedUNet

def create_model(
    config: CascadeConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    logger: Optional[logging.Logger] = None
) -> torch.nn.Module:
    """
    Create a model instance based on configuration.
    
    Args:
        config: Model configuration
        device: Device to place the model on
        logger: Logger for logging model information
        
    Returns:
        Initialized model
    """
    if logger:
        # Check if config is a TrainerConfig or a CascadeConfig
        if hasattr(config, 'cascade'):
            stages = config.cascade.stages
        else:
            stages = config.stages
            
        logger.info(f"Creating model with {len(stages)} stages")
        for i, stage in enumerate(stages):
            logger.info(f"Stage {i+1}: {stage.input_classes} -> {stage.target_class}")
            logger.info(f"  Encoder type: {stage.encoder_type}")
            
    # Create the model instance - Handle both TrainerConfig and CascadeConfig
    if hasattr(config, 'cascade'):
        model = FlexibleCascadedUNet(config.cascade)
    else:
        model = FlexibleCascadedUNet(config)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if logger:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    if logger:
        logger.info(f"Model created and moved to {device}")
        
    return model

def initialize_model_optimizer(
    config: CascadeConfig,
    logger: Optional[logging.Logger] = None,
    model: Optional[torch.nn.Module] = None,
    load_best: bool = True
) -> tuple:
    """
    Initialize model and optimizer based on configuration.
    If model is not provided, it will be created.
    
    Args:
        config: Training configuration
        logger: Optional logger
        model: Optional pre-created model (will be created if None)
        load_best: Whether to automatically load the best checkpoint if available
        
    Returns:
        Tuple of (model, optimizer, start_epoch, best_val_loss, metrics_history)
    """
    if logger:
        logger.info("Setting up model and optimizer...")
    
    # Create model if not provided
    if model is None:
        model = create_model(
            config=config,
            device=config.training.device if hasattr(config.training, 'device') else "cuda" if torch.cuda.is_available() else "cpu",
            logger=logger
        )
        
    # Create optimizer based on config type
    if config.training.optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.training.optimizer_type}")
    
    # Initialize training state variables
    start_epoch = 0
    best_val_loss = float('inf')
    metrics_history = {}
    
    # Load checkpoint if available and requested
    if load_best:
        checkpoint_dir = Path(config.save_dir) / config.experiment_name / "checkpoints"
        if checkpoint_dir.exists():
            # Look for the best checkpoint file
            best_checkpoint = checkpoint_dir / "model_best.pth"
            if best_checkpoint.exists():
                if logger:
                    logger.info("=" * 50)
                    logger.info(f"Found existing checkpoint at {best_checkpoint}")
                
                try:
                    # Load checkpoint to verify architecture
                    checkpoint = torch.load(best_checkpoint, map_location='cpu')
                    
                    # Verify if model architecture is compatible by checking the config
                    if 'config' in checkpoint:
                        saved_config = checkpoint['config']
                        architecture_match = True
                        
                        # Check cascade stages
                        if hasattr(config, 'cascade') and hasattr(saved_config, 'cascade'):
                            if len(config.cascade.stages) != len(saved_config.cascade.stages):
                                architecture_match = False
                                if logger:
                                    logger.warning("Number of stages mismatch")
                            else:
                                # Verify each stage configuration
                                for i, (current_stage, saved_stage) in enumerate(zip(config.cascade.stages, saved_config.cascade.stages)):
                                    if (current_stage.encoder_type != saved_stage.encoder_type or 
                                        current_stage.num_layers != saved_stage.num_layers or 
                                        current_stage.target_class != saved_stage.target_class):
                                        architecture_match = False
                                        if logger:
                                            logger.warning(f"Stage {i+1} architecture mismatch:")
                                            logger.warning(f"  Current: {current_stage.encoder_type}/"
                                                         f"{current_stage.num_layers}/{current_stage.target_class}")
                                            logger.warning(f"  Saved: {saved_stage.encoder_type}/"
                                                         f"{saved_stage.num_layers}/{saved_stage.target_class}")
                                        break
                        
                        if not architecture_match:
                            if logger:
                                logger.warning("Found checkpoint but model architecture doesn't match")
                                logger.warning("Starting with fresh model")
                                logger.info("=" * 50)
                            return model, optimizer, start_epoch, best_val_loss, metrics_history
                    
                    # Load the weights and optimizer state
                    if logger:
                        logger.info("Loading checkpoint state:")
                        
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                    best_val_loss = checkpoint['best_val_loss']
                    
                    if 'metrics_history' in checkpoint:
                        metrics_history = checkpoint['metrics_history']
                        
                    if logger:
                        logger.info(f"  Resuming from epoch {checkpoint['epoch']}")
                        logger.info(f"  Best validation loss: {best_val_loss:.4f}")
                        if metrics_history:
                            last_metrics = metrics_history.get('val_metrics', [])[-1] if metrics_history.get('val_metrics') else {}
                            for key, value in last_metrics.items():
                                if isinstance(value, (float, int)):
                                    logger.info(f"  Last {key}: {value:.4f}")
                        logger.info("=" * 50)
                
                except Exception as e:
                    if logger:
                        logger.error(f"Error loading checkpoint: {str(e)}")
                        logger.warning("Initializing from scratch due to checkpoint loading error")
                        logger.info("=" * 50)
        elif logger:
            logger.info("No existing checkpoints found. Initializing from scratch.")
            logger.info("=" * 50)
    elif logger:
        logger.info("Checkpoint loading disabled. Initializing from scratch.")
        logger.info("=" * 50)
    
    return model, optimizer, start_epoch, best_val_loss, metrics_history