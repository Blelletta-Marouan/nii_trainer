from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import numpy as np
from torch.utils.data import DataLoader

from ..configs.config import TrainerConfig
from ..visualization.visualizer import SegmentationVisualizer

def setup_visualizer(
    class_names: List[str],
    save_dir: Union[str, Path] = "visualizations",
    experiment_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> SegmentationVisualizer:
    """
    Set up a visualizer for segmentation results.
    
    Args:
        class_names: List of class names
        save_dir: Directory to save visualizations
        experiment_name: Name of the experiment (for subdirectory)
        logger: Logger for logging information
        
    Returns:
        Configured visualizer instance
    """
    if logger:
        logger.info("Setting up visualizer...")
    
    # Create save directory with experiment name if provided
    save_path = Path(save_dir)
    if experiment_name:
        save_path = save_path / experiment_name
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info(f"Visualizations will be saved to {save_path}")
    
    # Create visualizer
    visualizer = SegmentationVisualizer(
        class_names=class_names,
        save_dir=str(save_path)
    )
    
    return visualizer

def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    visualizer: SegmentationVisualizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = 4,
    save_dir: Union[str, Path] = "predictions",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Generate and save visualizations of model predictions.
    
    Args:
        model: Model to generate predictions
        dataloader: DataLoader containing samples to visualize
        visualizer: Visualizer instance
        device: Device to run the model on
        max_samples: Maximum number of samples to visualize
        save_dir: Directory to save visualizations
        logger: Logger for logging information
    """
    if logger:
        logger.info(f"Generating prediction visualizations for {max_samples} samples...")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Get predictions based on model type
            if hasattr(model, "get_predictions"):
                predictions = model.get_predictions(outputs)
            else:
                # Handle models without get_predictions method
                predictions = torch.argmax(outputs, dim=1)
            
            # Visualize batch
            visualizer.visualize_batch(
                images=images.cpu(),
                predictions=predictions.cpu(),
                targets=targets,
                max_samples=max_samples,
                save_dir=str(save_path)
            )
            
            # Generate and save confusion matrix
            visualizer.plot_confusion_matrix(
                predictions=predictions.cpu(),
                targets=targets,
                save_path=str(save_path / "confusion_matrix.png")
            )
            
            if logger:
                logger.info(f"Saved visualizations to {save_path}")
            
            # Only visualize one batch
            break

def visualize_metrics(
    trainer,
    visualizer: SegmentationVisualizer,
    save_path: Union[str, Path] = "metrics.png",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Generate and save visualization of training metrics.
    
    Args:
        trainer: Trainer instance with metrics history
        visualizer: Visualizer instance
        save_path: Path to save the metrics visualization
        logger: Logger for logging information
    """
    if logger:
        logger.info("Generating metrics visualization...")
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate metrics plot
    visualizer.plot_metrics(
        metrics=trainer.metrics_history,
        save_path=str(save_path)
    )
    
    if logger:
        logger.info(f"Saved metrics visualization to {save_path}")

def generate_visualizations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    trainer=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    base_dir: Union[str, Path] = "visualizations",
    experiment_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Generate all visualizations for a trained model.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing samples to visualize
        class_names: List of class names
        trainer: Optional trainer instance for metrics visualization
        device: Device to run the model on
        base_dir: Base directory for all visualizations
        experiment_name: Experiment name for subdirectory
        logger: Logger for logging information
    """
    # Create base directory
    base_path = Path(base_dir)
    if experiment_name:
        base_path = base_path / experiment_name
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Set up visualizer
    visualizer = setup_visualizer(
        class_names=class_names,
        save_dir=base_path,
        logger=logger
    )
    
    # Create predictions directory
    pred_dir = base_path / "predictions"
    
    # Visualize predictions
    visualize_predictions(
        model=model,
        dataloader=dataloader,
        visualizer=visualizer,
        device=device,
        save_dir=pred_dir,
        logger=logger
    )
    
    # Visualize metrics if trainer is provided
    if trainer and hasattr(trainer, 'metrics_history'):
        metrics_path = base_path / "metrics.png"
        visualize_metrics(
            trainer=trainer,
            visualizer=visualizer,
            save_path=metrics_path,
            logger=logger
        )
    
    if logger:
        logger.info(f"All visualizations saved to {base_path}")