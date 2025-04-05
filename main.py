#!/usr/bin/env python3
"""
Main module for NII Trainer - A modular framework for training cascaded neural networks
on medical imaging data (NIfTI format).

This module provides easy-to-use functions for running segmentation experiments with NII files
in a Jupyter notebook or Python script environment.
"""

from pathlib import Path
import logging

from nii_trainer.configs.config import (
    TrainerConfig,
    StageConfig,
    CascadeConfig,
    DataConfig,
    TrainingConfig,
    LossConfig
)
from nii_trainer.utils import Experiment, setup_logger
from nii_trainer.models import (
    MetricsManager,
    GradientManager,
    VisualizationManager,
    CurriculumManager,
    ModelTrainer
)

def create_liver_tumor_config(
    volume_dir="volume_pt1",
    output_dir="processed_data",
    img_size=(512, 512),
    batch_size=16,
    window_width=180,
    window_level=50,
    skip_empty=False,
    slice_step=1,
    train_val_test_split=(0.6, 0.2, 0.2),
    learning_rate=1e-4,
    epochs=100,
    batch_accumulation=1,
    experiment_name="liver_tumor_cascade",
):
    """
    Create a default configuration for liver-tumor segmentation.
    
    Args:
        volume_dir: Directory containing volume files
        output_dir: Directory to save processed data
        img_size: Target image size (width, height)
        batch_size: Batch size for training
        window_width: CT window width
        window_level: CT window level
        skip_empty: Whether to skip slices without annotations
        slice_step: Take every nth slice
        train_val_test_split: Proportions for train/val/test split
        learning_rate: Initial learning rate
        epochs: Maximum number of training epochs
        batch_accumulation: Gradient accumulation steps
        experiment_name: Name of experiment
        
    Returns:
        Complete TrainerConfig for liver-tumor segmentation
    """
    config = TrainerConfig(
        data=DataConfig(
            base_dir=str(Path(volume_dir).absolute()),
            output_dir=output_dir,
            classes=["background", "liver", "tumor"],
            class_map={"background": 0, "liver": 1, "tumor": 2},
            img_size=img_size,
            batch_size=batch_size,
            window_params={"window_width": window_width, "window_level": window_level},
            skip_empty=skip_empty,
            slice_step=slice_step,
            train_val_test_split=train_val_test_split
        ),
        cascade=CascadeConfig(
            stages=[
                # Stage 1: Binary liver segmentation
                StageConfig(
                    input_classes=["background", "liver"],
                    target_class="liver",
                    encoder_type="efficientnet",
                    num_layers=3,
                    is_binary=True,
                    threshold=0.5
                ),
                # Stage 2: Tumor segmentation within liver
                StageConfig(
                    input_classes=["liver"],
                    target_class="tumor",
                    encoder_type="efficientnet",
                    num_layers=4,
                    is_binary=True,
                    threshold=0.5
                )
            ],
            in_channels=1,
            initial_features=32,
            feature_growth=2.0,
            pretrained=True
        ),
        training=TrainingConfig(
            learning_rate=learning_rate,
            epochs=epochs,
            patience=10,
            mixed_precision=True,
            batch_accumulation=batch_accumulation
        ),
        loss=LossConfig(
            stage_weights=[1.0, 1.5],  # Higher weight for tumor stage
            class_weights={
                "background": 1.0,
                "liver": 2.0,
                "tumor": 4.0  # Higher weight for tumor class
            }
        ),
        experiment_name=experiment_name
    )
    
    return config

def get_default_curriculum_params():
    """Get default curriculum learning parameters for cascaded network training."""
    return {
        'stage_schedule': [
            (0, 10),  # Train liver stage for 50 epochs
            (10, 30)   # Train tumor stage for 50 more epochs
        ],
        'learning_rates': [1e-3, 5e-4],  # Higher LR for first stage, lower for second
        'stage_freezing': [False, True]   # Second stage freezes the first stage
    }

def run_liver_tumor_segmentation(
    volume_dir="volume_pt1",
    segmentation_dir="segmentations",
    output_dir="experiments",
    experiment_name="liver_tumor_cascade",
    process_data=True,
    force_overwrite=False,
    use_curriculum=True,
    img_size=(512, 512),
    batch_size=32,
    window_width=180,
    window_level=50,
    slice_step=1,
    skip_empty=False,
    learning_rate=1e-4,
    epochs=30,
    batch_accumulation=1,
    custom_config=None,
    custom_curriculum_params=None,
    verbose=True
):
    """
    Run a complete liver-tumor segmentation experiment.
    
    This function handles the entire pipeline from data preprocessing to model training
    and evaluation. It's designed to be easy to use in Jupyter notebooks or scripts.
    
    Args:
        volume_dir: Directory containing volume NIfTI files
        segmentation_dir: Directory containing segmentation NIfTI files
        output_dir: Base directory for experiment outputs
        experiment_name: Name of the experiment
        process_data: Whether to process the NIfTI data
        force_overwrite: Whether to overwrite existing processed data
        use_curriculum: Whether to use curriculum learning
        img_size: Target image size (width, height)
        batch_size: Batch size for training
        window_width: CT window width
        window_level: CT window level
        slice_step: Take every nth slice
        skip_empty: Whether to skip slices without annotations
        learning_rate: Initial learning rate
        epochs: Maximum number of training epochs
        batch_accumulation: Number of batches to accumulate gradients over
        custom_config: Optional custom TrainerConfig (overrides all other config params)
        custom_curriculum_params: Optional custom curriculum parameters
        verbose: Whether to print detailed logs
        
    Returns:
        Experiment instance with trained model and results
    """
    # Setup logger
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logger(
        experiment_name=experiment_name,
        save_dir="logs",
        level=log_level
    )
    
    logger.info("=== Starting NII Trainer Pipeline ===")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Volume directory: {volume_dir}")
    logger.info(f"Segmentation directory: {segmentation_dir}")
    logger.info(f"Using curriculum learning: {use_curriculum}")
    
    # Use custom config if provided, otherwise create default
    config = custom_config
    if config is None:
        config = create_liver_tumor_config(
            volume_dir=volume_dir,
            img_size=img_size,
            batch_size=batch_size,
            window_width=window_width,
            window_level=window_level,
            skip_empty=skip_empty,
            slice_step=slice_step,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_accumulation=batch_accumulation,
            experiment_name=experiment_name
        )
    
    # Add a use_curriculum flag to the config for clarity
    if not hasattr(config, 'use_curriculum'):
        config.use_curriculum = use_curriculum
    
    # Get curriculum parameters
    curriculum_params = custom_curriculum_params
    if use_curriculum and curriculum_params is None:
        curriculum_params = get_default_curriculum_params()
    elif not use_curriculum:
        # Ensure curriculum is disabled when use_curriculum=False
        curriculum_params = None
    
    # Create and run experiment
    experiment = Experiment(
        config=config,
        experiment_name=experiment_name,
        base_dir=output_dir,
        logger=logger
    )
    
    # Run the complete pipeline
    results = experiment.run(
        volume_dir=volume_dir,
        segmentation_dir=segmentation_dir,
        process_data=process_data,
        curriculum=use_curriculum,  # Pass the flag to experiment.run
        curriculum_params=curriculum_params,  # This will be None if use_curriculum=False
        force_overwrite=force_overwrite
    )
    
    logger.info("=== NII Trainer Pipeline completed successfully ===")
    
    return experiment

def create_custom_experiment(config, **kwargs):
    """
    Create a custom experiment with a specified configuration.
    
    Args:
        config: TrainerConfig object with experiment configuration
        **kwargs: Additional parameters to pass to Experiment constructor
        
    Returns:
        Configured Experiment instance (not run yet)
    """
    experiment = Experiment(config=config, **kwargs)
    return experiment