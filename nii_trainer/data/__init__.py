"""
Data preprocessing and handling utilities for NIfTI images.
"""

from .preprocessing import NiiPreprocessor
from .dataset import MultiClassSegDataset, PairedTransform
from .batch_processor import BatchProcessor
from .nifti_processor import (
    read_nifti_file,
    apply_window,
    preprocess_volume,
    process_volume_and_segmentation,
    create_directory_structure
)
from .data_utils import (
    create_data_transforms,
    create_datasets,
    create_dataloaders,
    setup_data_pipeline
)

__all__ = [
    # Legacy imports for backward compatibility
    'NiiPreprocessor',
    'read_nii',
    'apply_window',
    'MultiClassSegDataset',
    'PairedTransform',
    'BatchProcessor',
    
    # New modular functions for nifti processing
    'read_nifti_file',
    'preprocess_volume',
    'process_volume_and_segmentation',
    'create_directory_structure',
    
    # New modular functions for data pipeline
    'create_data_transforms',
    'create_datasets', 
    'create_dataloaders',
    'setup_data_pipeline'
]

# For backward compatibility
read_nii = read_nifti_file