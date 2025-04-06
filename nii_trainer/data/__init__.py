"""
Data processing and handling utilities for medical imaging data.

This module provides comprehensive tools for:
- Processing 3D medical imaging data (NIfTI format)
- Batch processing of volume-segmentation pairs
- Dataset creation and management
- Data augmentation and preprocessing
"""

# Core data processing
from .nifti_processor import (
    read_nifti_file,
    apply_window,
    preprocess_volume,
    process_volume_and_segmentation,
    create_directory_structure
)

# Dataset and transforms
from .dataset import (
    MultiClassSegDataset,
    PairedTransform,
    create_dataloader
)

# Batch processing
from .batch_processor import BatchProcessor

# Pipeline utilities
from .data_utils import (
    create_data_transforms,
    create_datasets,
    create_dataloaders,
    setup_data_pipeline
)

__all__ = [
    # Core processing
    'read_nifti_file',
    'apply_window',
    'preprocess_volume',
    'process_volume_and_segmentation',
    'create_directory_structure',
    
    # Dataset components
    'MultiClassSegDataset',
    'PairedTransform',
    'create_dataloader',
    
    # Batch processing
    'BatchProcessor',
    
    # Pipeline utilities
    'create_data_transforms',
    'create_datasets',
    'create_dataloaders',
    'setup_data_pipeline'
]

# Remove deprecated imports
import warnings
warnings.warn(
    "The NiiPreprocessor class is deprecated and will be removed in a future version. "
    "Use BatchProcessor or individual processing functions instead.",
    DeprecationWarning,
    stacklevel=2
)