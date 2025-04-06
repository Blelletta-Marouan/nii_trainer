"""
NII Trainer Package - A flexible framework for training neural networks on NIfTI data.

This package provides a modular and configurable framework for:
- Processing 3D medical imaging data in NIfTI format
- Training cascaded neural networks for segmentation
- Curriculum learning with multi-stage training
- Comprehensive metrics tracking and visualization
"""

from .configs.config import (
    TrainerConfig, DataConfig, ModelConfig, 
    CascadeConfig, LossConfig, TrainingConfig, 
    StageConfig, VisualizationConfig, CurriculumConfig
)

from .models import (
    FlexibleCascadedUNet,
    ModelTrainer,
    create_model,
    initialize_model_optimizer,
    setup_trainer
)

from .data import (
    process_volume_and_segmentation,
    setup_data_pipeline,
    BatchProcessor,
    MultiClassSegDataset
)

from .utils.experiment import Experiment
from .utils.logging_utils import setup_logger
from .utils.metrics import calculate_class_metrics, calculate_volumetric_metrics

__version__ = "0.1.0"
__author__ = "Marouan AI"

__all__ = [
    # Configurations
    'TrainerConfig',
    'DataConfig',
    'ModelConfig',
    'CascadeConfig',
    'LossConfig',
    'TrainingConfig',
    'StageConfig',
    'VisualizationConfig',
    'CurriculumConfig',
    
    # Models
    'FlexibleCascadedUNet',
    'ModelTrainer',
    'create_model',
    'initialize_model_optimizer',
    'setup_trainer',
    
    # Data handling
    'BatchProcessor',
    'MultiClassSegDataset',
    'setup_data_pipeline',
    'process_volume_and_segmentation',
    
    # Utilities
    'Experiment',
    'setup_logger',
    'calculate_class_metrics',
    'calculate_volumetric_metrics'
]