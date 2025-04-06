#!/usr/bin/env python3
"""
Main module for NII Trainer - A modular framework for training cascaded neural networks
on medical imaging data (NIfTI format).
"""

from pathlib import Path
import logging
import os
import sys
from typing import Optional, Dict, Any, List

# Import directly from the package
from nii_trainer import (
    TrainerConfig, StageConfig, CascadeConfig, DataConfig,
    TrainingConfig, LossConfig, CurriculumConfig, 
    Experiment, setup_logger
)

def create_segmentation_config(
    volume_dir: str,
    output_dir: str,
    classes: List[str],
    stages: List[Dict[str, Any]],
    experiment_name: str,
    **kwargs
) -> TrainerConfig:
    """
    Create a flexible segmentation configuration.
    
    Args:
        volume_dir: Directory containing volume files
        output_dir: Directory to save processed data
        classes: List of class names in order
        stages: List of stage configurations
        experiment_name: Name of experiment
        **kwargs: Additional configuration overrides
    """
    # Create class mapping
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    
    # Create stage configs
    stage_configs = []
    for stage_dict in stages:
        stage_configs.append(StageConfig(**stage_dict))
    
    # Extract configuration parameters with defaults
    data_params = kwargs.get('data', {})
    training_params = kwargs.get('training', {})
    loss_params = kwargs.get('loss', {})
    curriculum_params = kwargs.get('curriculum', {})
    
    # Create the complete configuration
    config = TrainerConfig(
        data=DataConfig(
            base_dir=str(Path(volume_dir).absolute()),
            output_dir=output_dir,
            classes=classes,
            class_map=class_map,
            **data_params
        ),
        cascade=CascadeConfig(
            stages=stage_configs,
            in_channels=kwargs.get('in_channels', 1),
            initial_features=kwargs.get('initial_features', 32),
            feature_growth=kwargs.get('feature_growth', 2.0),
            pretrained=kwargs.get('pretrained', True)
        ),
        training=TrainingConfig(**training_params),
        loss=LossConfig(**loss_params),
        experiment_name=experiment_name
    )
    
    # If curriculum parameters are provided, ensure they're properly set in the config
    if curriculum_params:
        config.training.curriculum.enabled = curriculum_params.get('enabled', False)
        if 'stage_schedule' in curriculum_params:
            config.training.curriculum.stage_schedule = curriculum_params['stage_schedule']
        if 'learning_rates' in curriculum_params:
            config.training.curriculum.learning_rates = curriculum_params['learning_rates']
        if 'stage_freezing' in curriculum_params:
            config.training.curriculum.stage_freezing = curriculum_params['stage_freezing']
        if 'stage_overlap' in curriculum_params:
            config.training.curriculum.stage_overlap = curriculum_params['stage_overlap']
        if 'stage_metrics' in curriculum_params:
            config.training.curriculum.stage_metrics = curriculum_params['stage_metrics']
        if 'use_gradual_unfreezing' in curriculum_params:
            config.training.curriculum.use_gradual_unfreezing = curriculum_params['use_gradual_unfreezing']
        if 'final_joint_training' in curriculum_params:
            config.training.curriculum.final_joint_training = curriculum_params['final_joint_training']
        if 'warm_up_epochs' in curriculum_params:
            config.training.curriculum.warm_up_epochs = curriculum_params['warm_up_epochs']
        if 'curriculum_metrics' in curriculum_params:
            config.training.curriculum.curriculum_metrics = curriculum_params['curriculum_metrics']
    
    return config

def create_liver_tumor_example():
    """Create an example liver-tumor segmentation configuration."""
    return create_segmentation_config(
        volume_dir="volume_pt1",
        output_dir="data",  # Updated to a better default
        classes=["background", "liver", "tumor"],
        stages=[
            {
                "input_classes": ["background", "liver"],
                "target_class": "liver",
                "encoder_type": "efficientnet",
                "num_layers": 3,
                "is_binary": True
            },
            {
                "input_classes": ["liver"],
                "target_class": "tumor",
                "encoder_type": "efficientnet",
                "num_layers": 4,
                "is_binary": True
            }
        ],
        experiment_name="liver_tumor_seg",
        data={
            "img_size": (512, 512),
            "batch_size": 16,
            "slice_step": 1,
            "skip_empty": True,
            "window_params": {
                "window_width": 180,
                "window_level": 50
            },
            "augmentation_params": {
                "enabled": True,
                "spatial": {
                    "rotation_range": (-15, 15),
                    "scale_range": (0.85, 1.15),
                    "flip_probability": 0.5
                },
            }
        },
        training={
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_accumulation": 8,
            "mixed_precision": True,
            "scheduler_type": "plateau",
            "scheduler_params": {
                "patience": 5,
                "factor": 0.5
            }
        },
        loss={
            "stage_weights": [1.0, 1.5],
            "class_weights": {
                "background": 1.0,
                "liver": 2.0,
                "tumor": 4.0
            }
        },
        curriculum={
            "enabled": True,
            "stage_schedule": [(0, 5), (1, 15)],
            "learning_rates": [1e-3, 5e-4],
            "stage_freezing": [False, True],
            "stage_overlap": 5  # Number of epochs to overlap between stages
        }
    )

def run_segmentation(
    config: TrainerConfig,
    volume_dir: str,
    segmentation_dir: str,
    process_data: bool = True,
    force_overwrite: bool = False,
    verbose: bool = True,
    curriculum: bool = True,
    curriculum_params: Optional[Dict[str, Any]] = None,
    load_best: bool = True
) -> Experiment:
    """
    Run a complete segmentation experiment with the given configuration.
    
    Args:
        config: Complete trainer configuration
        volume_dir: Directory containing volume files
        segmentation_dir: Directory containing segmentation files
        process_data: Whether to process the NIfTI data
        force_overwrite: Whether to overwrite existing processed data
        verbose: Whether to print detailed logs
        curriculum: Whether to use curriculum learning
        curriculum_params: Optional parameters for curriculum learning
        load_best: Whether to automatically load the best checkpoint if the folder contains 
                  the same architecture as the model we're trying to initialize
    
    Returns:
        Experiment instance with results
    """
    # Setup logging
    logger = setup_logger(
        experiment_name=config.experiment_name,
        save_dir=str(Path(config.save_dir) / "logs"),
        level=logging.INFO if verbose else logging.WARNING
    )
    
    logger.info("=== Starting NII Trainer Pipeline ===")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Classes: {config.data.classes}")
    logger.info(f"Number of stages: {len(config.cascade.stages)}")
    logger.info(f"Auto-load best checkpoint: {load_best}")
    
    # Create and run experiment
    experiment = Experiment(
        config=config,
        logger=logger
    )
    
    # Run complete pipeline
    results = experiment.run(
        volume_dir=volume_dir,
        segmentation_dir=segmentation_dir,
        process_data=process_data,
        force_overwrite=force_overwrite,
        curriculum=curriculum,
        curriculum_params=curriculum_params,
        load_best=load_best
    )
    
    logger.info("=== Pipeline completed successfully ===")
    
    return experiment

if __name__ == "__main__":
    # Example usage
    config = create_liver_tumor_example()
    
    # Get absolute paths to data directories
    base_dir = Path(__file__).parent.absolute()
    volume_dir = str(base_dir / "volume_pt1")
    segmentation_dir = str(base_dir / "segmentations")
    
    print(f"Using volume directory: {volume_dir}")
    print(f"Using segmentation directory: {segmentation_dir}")
    
    # Run the experiment
    experiment = run_segmentation(
        config=config,
        volume_dir=volume_dir,
        segmentation_dir=segmentation_dir,
        process_data=True,
        force_overwrite=False,
        verbose=True
    )
    
    print(f"Experiment completed: {experiment.config.experiment_name}")
    print(f"Results and models saved to: {Path(experiment.config.save_dir) / experiment.config.experiment_name}")