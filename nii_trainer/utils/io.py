"""
I/O utilities for NII-Trainer.
"""

import os
import json
import pickle
import shutil
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union
from ..core.exceptions import NIITrainerError


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save model checkpoint with training state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'metadata': metadata or {}
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint and restore training state."""
    if not os.path.exists(filepath):
        raise NIITrainerError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def save_predictions(
    predictions: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'pickle'
) -> None:
    """Save model predictions to file."""
    filepath = Path(filepath)
    create_directory(filepath.parent)
    
    if format.lower() == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(predictions, f)
    elif format.lower() == 'json':
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_predictions(
    filepath: Union[str, Path],
    format: str = 'pickle'
) -> Dict[str, Any]:
    """Load model predictions from file."""
    if not os.path.exists(filepath):
        raise NIITrainerError(f"Predictions file not found: {filepath}")
    
    if format.lower() == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format.lower() == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_directory(path: Union[str, Path], exist_ok: bool = True) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=exist_ok)


def get_file_size(filepath: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy file from source to destination."""
    shutil.copy2(src, dst)


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Move file from source to destination."""
    shutil.move(src, dst)


def cleanup_directory(
    directory: Union[str, Path],
    pattern: str = "*",
    older_than_days: Optional[int] = None
) -> int:
    """Clean up directory by removing files matching pattern."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    files_removed = 0
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if older_than_days is None:
                file_path.unlink()
                files_removed += 1
            else:
                import time
                file_age = time.time() - file_path.stat().st_mtime
                if file_age > (older_than_days * 24 * 3600):
                    file_path.unlink()
                    files_removed += 1
    
    return files_removed