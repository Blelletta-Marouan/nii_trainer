"""Model architectures and training components."""

# Core model architectures
from .cascaded_unet import FlexibleCascadedUNet
from .base import EncoderFactory, EncoderBase, MobileNetV2Encoder, ResNetEncoder, EfficientNetEncoder

# Training components
from .model_trainer import ModelTrainer
from .training_loop import TrainingLoop
from .early_stopping import EarlyStoppingHandler
from .checkpoint_manager import CheckpointManager

# Management components
from .metrics_manager import MetricsManager, MetricAggregator
from .visualization_manager import VisualizationManager
from .curriculum_manager import CurriculumManager
from .gradient_manager import GradientManager

# Utilities
from .model_utils import create_model, initialize_model_optimizer
from .training_utils import setup_trainer

__all__ = [
    # Model architectures
    'FlexibleCascadedUNet',
    'EncoderFactory',
    'EncoderBase',
    'MobileNetV2Encoder',
    'ResNetEncoder',
    'EfficientNetEncoder',
    
    # Training components
    'ModelTrainer',
    'TrainingLoop',
    'EarlyStoppingHandler',
    'CheckpointManager',
    
    # Management components
    'MetricsManager',
    'MetricAggregator',
    'VisualizationManager',
    'CurriculumManager',
    'GradientManager',
    
    # Utilities
    'create_model',
    'initialize_model_optimizer',
    'setup_trainer'
]