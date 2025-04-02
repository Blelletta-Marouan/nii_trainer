from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import torch
import os

from ..configs.config import TrainerConfig
from ..data import (
    process_volume_and_segmentation,
    setup_data_pipeline
)
from ..models import (
    initialize_training_components,
    setup_trainer,
    train_model,
    train_with_curriculum,
    evaluate_model
)
from ..visualization import (
    generate_visualizations
)
from ..utils.logging_utils import setup_logger

class Experiment:
    """
    High-level class to manage complete training experiments.
    
    This class orchestrates the complete pipeline from data preprocessing
    to model training and evaluation.
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        experiment_name: Optional[str] = None,
        base_dir: Union[str, Path] = "experiments",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize an experiment.
        
        Args:
            config: Trainer configuration
            experiment_name: Name of the experiment (for directory structure)
            base_dir: Base directory for experiment outputs
            logger: Logger for logging information
        """
        self.config = config
        self.experiment_name = experiment_name or config.experiment_name
        self.base_dir = Path(base_dir) / self.experiment_name
        
        # Set up directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_dir / "data"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.visualizations_dir = self.base_dir / "visualizations"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logger or setup_logger(
            experiment_name=self.experiment_name,
            save_dir=str(self.base_dir / "logs")
        )
        
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(f"Base directory: {self.base_dir}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.datasets = None
        self.dataloaders = None
        
    def process_data(
        self,
        volume_dir: Union[str, Path],
        segmentation_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        include_val: bool = True,
        include_test: bool = True,
        force_overwrite: bool = False,
        **kwargs
    ):
        """
        Process NIfTI data for the experiment.
        
        Args:
            volume_dir: Directory containing volume files
            segmentation_dir: Directory containing segmentation files
            output_dir: Directory to save processed data (defaults to experiment data dir)
            include_val: Whether to create validation split
            include_test: Whether to create test split
            force_overwrite: Whether to overwrite existing data
            **kwargs: Additional arguments to pass to processor
        """
        output_dir = output_dir or self.data_dir
        self.config.data.output_dir = str(output_dir)
        
        self.logger.info("Processing NIfTI data for experiment...")
        
        # Check if data already exists
        if Path(output_dir).exists() and not force_overwrite:
            train_dir = Path(output_dir) / "train" / "data"
            if train_dir.exists() and any(train_dir.iterdir()):
                self.logger.info("Processed data already exists, skipping processing")
                return
        
        # Process data
        start_time = time.time()
        
        # Extract parameters from config
        window_params = getattr(self.config.data, 'window_params', None)
        slice_step = getattr(self.config.data, 'slice_step', 1)
        skip_empty = getattr(self.config.data, 'skip_empty', False)
        train_val_test_split = getattr(self.config.data, 'train_val_test_split', (0.7, 0.15, 0.15))
        img_size = getattr(self.config.data, 'img_size', (512, 512))
        
        # Create output directory by experiment
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        from ..data import BatchProcessor
        processor = BatchProcessor(
            img_size=img_size,
            window_params=window_params,
            skip_empty=skip_empty,
            slice_step=slice_step,
            train_val_test_split=train_val_test_split,
            logger=self.logger
        )
        
        processed_pairs = processor.process_batch(
            volume_dir=volume_dir,
            segmentation_dir=segmentation_dir,
            output_dir=output_dir,
            include_val=include_val,
            include_test=include_test,
            force_overwrite=force_overwrite,
            **kwargs
        )
        
        processing_time = time.time() - start_time
        self.logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Total processed image pairs: {len(processed_pairs) if processed_pairs else 0}")
        
    def setup_data_pipeline(self):
        """
        Set up data pipeline for the experiment.
        """
        self.logger.info("Setting up data pipeline...")
        
        # Create datasets and dataloaders
        self.datasets, self.dataloaders = setup_data_pipeline(
            config=self.config.data,
            logger=self.logger
        )
        
        return self.datasets, self.dataloaders
    
    def setup_model(self):
        """
        Set up model, optimizer, and trainer for the experiment.
        """
        self.logger.info("Setting up model and training components...")
        
        # Initialize model, optimizer, and load checkpoints
        self.model, self.optimizer, start_epoch, best_val_loss, metrics_history = (
            initialize_training_components(
                config=self.config,
                logger=self.logger
            )
        )
        
        # Set up trainer
        self.trainer = setup_trainer(
            config=self.config,
            model=self.model,
            train_loader=self.dataloaders["train"],
            val_loader=self.dataloaders["val"] if "val" in self.dataloaders else None,
            optimizer=self.optimizer,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            metrics_history=metrics_history,
            logger=self.logger
        )
        
        return self.model, self.trainer
    
    def train(self, curriculum: bool = False, curriculum_params: Optional[Dict[str, Any]] = None):
        """
        Train the model.
        
        Args:
            curriculum: Whether to use curriculum learning
            curriculum_params: Parameters for curriculum learning
        """
        if self.trainer is None:
            self.logger.warning("Trainer not set up. Run setup_model() first.")
            return
        
        self.logger.info("Starting model training...")
        
        # Train the model
        if curriculum:
            if curriculum_params is None:
                curriculum_params = {}
                
            # Default curriculum parameters
            stage_schedule = curriculum_params.get('stage_schedule', [
                (0, 50),  # Train first stage for 50 epochs
                (1, 50)   # Train second stage for 50 epochs
            ])
            
            learning_rates = curriculum_params.get('learning_rates', [
                self.config.training.learning_rate,
                self.config.training.learning_rate / 2
            ])
            
            stage_freezing = curriculum_params.get('stage_freezing', [False, True])
            
            self.logger.info("Using curriculum learning approach")
            self.logger.info(f"Stage schedule: {stage_schedule}")
            self.logger.info(f"Learning rates: {learning_rates}")
            self.logger.info(f"Stage freezing: {stage_freezing}")
            
            train_with_curriculum(
                trainer=self.trainer,
                stage_schedule=stage_schedule,
                learning_rates=learning_rates,
                stage_freezing=stage_freezing,
                checkpoint_dir=self.checkpoints_dir,
                logger=self.logger
            )
        else:
            train_model(
                trainer=self.trainer,
                config=self.config,
                checkpoint_dir=self.checkpoints_dir,
                logger=self.logger
            )
        
        self.logger.info("Model training completed")
        
    def evaluate(self):
        """
        Evaluate the trained model.
        """
        if self.trainer is None:
            self.logger.warning("Trainer not set up. Run setup_model() first.")
            return
        
        self.logger.info("Evaluating model...")
        
        # Evaluate on validation set
        val_metrics = evaluate_model(
            trainer=self.trainer,
            config=self.config,
            test_loader=self.dataloaders.get("val"),
            logger=self.logger
        )
        
        # Evaluate on test set if available
        if "test" in self.dataloaders:
            self.logger.info("Evaluating on test set...")
            test_metrics = evaluate_model(
                trainer=self.trainer,
                config=self.config,
                test_loader=self.dataloaders["test"],
                logger=self.logger
            )
        
        return val_metrics
    
    def visualize(self):
        """
        Generate visualizations for the experiment.
        """
        if self.model is None:
            self.logger.warning("Model not set up. Run setup_model() first.")
            return
        
        self.logger.info("Generating visualizations...")
        
        # Generate visualizations
        generate_visualizations(
            model=self.model,
            dataloader=self.dataloaders["val"] if "val" in self.dataloaders else self.dataloaders["train"],
            class_names=self.config.data.classes,
            trainer=self.trainer,
            device=self.config.training.device,
            base_dir=self.visualizations_dir,
            logger=self.logger
        )
        
        self.logger.info(f"Visualizations saved to {self.visualizations_dir}")
    
    def run(
        self,
        volume_dir: Union[str, Path],
        segmentation_dir: Union[str, Path],
        process_data: bool = True,
        curriculum: bool = False,
        curriculum_params: Optional[Dict[str, Any]] = None,
        force_overwrite: bool = False
    ):
        """
        Run complete experiment pipeline.
        
        Args:
            volume_dir: Directory containing volume files
            segmentation_dir: Directory containing segmentation files
            process_data: Whether to process data
            curriculum: Whether to use curriculum learning
            curriculum_params: Parameters for curriculum learning
            force_overwrite: Whether to overwrite existing data
        """
        start_time = time.time()
        
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Process data if requested
        if process_data:
            self.process_data(
                volume_dir=volume_dir,
                segmentation_dir=segmentation_dir,
                force_overwrite=force_overwrite
            )
        
        # Set up data pipeline
        self.setup_data_pipeline()
        
        # Set up model
        self.setup_model()
        
        # Train model
        self.train(curriculum=curriculum, curriculum_params=curriculum_params)
        
        # Evaluate model
        self.evaluate()
        
        # Generate visualizations
        self.visualize()
        
        total_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {total_time/60:.2f} minutes")
        
        return {
            "model": self.model,
            "trainer": self.trainer,
            "datasets": self.datasets,
            "dataloaders": self.dataloaders
        }