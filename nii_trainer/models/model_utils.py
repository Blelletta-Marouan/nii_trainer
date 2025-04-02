from pathlib import Path
import logging
import torch
from typing import Dict, Any, Optional, Tuple, Union
import os

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
        logger.info(f"Creating model with {len(config.stages)} stages")
        for i, stage in enumerate(config.stages):
            logger.info(f"Stage {i+1}: {stage.input_classes} -> {stage.target_class}")
            logger.info(f"  Encoder type: {stage.encoder_type}")
            
    # Create the model instance
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

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load model checkpoint if it exists.
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        optimizer: Optional optimizer to load state into
        logger: Logger for logging checkpoint information
        
    Returns:
        Tuple of (success_flag, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.info(f"No checkpoint found at {checkpoint_path}")
        return False, None
    
    if device is None:
        device = next(model.parameters()).device
    
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a full model or just a stage
    if 'model_state_dict' in checkpoint:
        # Full model checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if logger:
                logger.info("Loaded optimizer state from checkpoint")
                
        if logger:
            logger.info(f"Loaded complete model checkpoint from {checkpoint_path}")
        
        return True, checkpoint
    
    elif 'state_dict' in checkpoint and 'stage_idx' in checkpoint:
        # Stage-specific checkpoint
        stage_idx = checkpoint['stage_idx']
        if hasattr(model, 'stages') and stage_idx < len(model.stages):
            # Handle DataParallel wrapped model
            if isinstance(model, torch.nn.DataParallel):
                model.module.stages[stage_idx].load_state_dict(checkpoint['state_dict'])
            else:
                model.stages[stage_idx].load_state_dict(checkpoint['state_dict'])
                
            if logger:
                logger.info(f"Loaded checkpoint for stage {stage_idx+1} from {checkpoint_path}")
            
            return True, checkpoint
    
    # Unknown checkpoint format
    if logger:
        logger.warning(f"Unrecognized checkpoint format at {checkpoint_path}")
        
    return False, None

def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_val_loss: float = float('inf'),
    metrics_history: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save the checkpoint
        optimizer: Optional optimizer to save state
        epoch: Current epoch number
        best_val_loss: Best validation loss
        metrics_history: Training metrics history
        is_best: Whether this is the best model so far
        logger: Logger for logging checkpoint information
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
    if metrics_history is not None:
        checkpoint['metrics_history'] = metrics_history
        
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    if logger:
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path), "best_model.pth")
        torch.save(checkpoint, best_path)
        
        if logger:
            logger.info(f"Saved best model checkpoint to {best_path}")

def initialize_training_components(
    config,
    logger: Optional[logging.Logger] = None
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float, Dict[str, Any]]:
    """
    Initialize model, optimizer and load checkpoints if available.
    
    Args:
        config: Training configuration
        logger: Logger for logging information
        
    Returns:
        Tuple of (model, optimizer, start_epoch, best_val_loss, metrics_history)
    """
    # Create model
    model = create_model(config.cascade, config.training.device, logger)
    
    # Initialize tracking variables
    start_epoch = 0
    best_val_loss = float('inf')
    metrics_history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
    
    # Check for existing checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Try to load full model checkpoint first
    full_model_loaded = False
    full_model_path = checkpoint_dir / "best_model.pth"
    
    # Create optimizer
    if logger:
        logger.info("Setting up optimizer...")
        
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    if full_model_path.exists():
        if logger:
            logger.info(f"Found full model checkpoint at {full_model_path}")
            
        loaded, checkpoint = load_checkpoint(
            model, 
            str(full_model_path), 
            config.training.device,
            optimizer,
            logger
        )
        
        if loaded:
            full_model_loaded = True
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            if 'metrics_history' in checkpoint:
                metrics_history = checkpoint['metrics_history']
                
            if logger:
                logger.info(f"Will resume training from epoch {start_epoch}")
                logger.info(f"Loaded trainer state: best val loss: {best_val_loss:.4f}")
    
    # If full model not loaded, check for individual stage checkpoints
    if not full_model_loaded:
        # Check for stage checkpoints
        for i in range(len(config.cascade.stages)):
            stage_path = checkpoint_dir / f"stage_{i+1}_best.pth"
            if stage_path.exists():
                if logger:
                    logger.info(f"Found stage {i+1} checkpoint at {stage_path}")
                load_checkpoint(model, str(stage_path), config.training.device, logger=logger)
    
    return model, optimizer, start_epoch, best_val_loss, metrics_history