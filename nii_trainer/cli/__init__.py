"""
Command Line Interface for NII-Trainer.
"""

from .main import main
from .commands import (
    train_command,
    evaluate_command,
    predict_command,
    export_command,
    config_command,
    info_command
)

__all__ = [
    "main",
    "train_command",
    "evaluate_command", 
    "predict_command",
    "export_command",
    "config_command",
    "info_command"
]