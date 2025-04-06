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
from ..utils.logging_utils import setup_logger

# Avoid circular imports by using late/deferred imports when needed
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
    
    def setup_model(self, load_best: bool = True):
        """
        Set up model, optimizer, and trainer for the experiment.
        
        Args:
            load_best: Whether to automatically load the best checkpoint if available
        """
        self.logger.info("Setting up model and training components...")
        
        # Use late import to avoid circular dependency
        from ..models.model_trainer import ModelTrainer
        from ..models.model_utils import initialize_model_optimizer         
        # Initialize model, optimizer, and load checkpoints
        self.model, self.optimizer, start_epoch, best_val_loss, metrics_history = (
            initialize_model_optimizer(
                config=self.config,
                logger=self.logger,
                load_best=load_best  # Pass the load_best parameter
            )
        )
        
        # Create scheduler if enabled in config
        scheduler = None
        if hasattr(self.config.training, 'use_scheduler') and self.config.training.use_scheduler:
            self.logger.info("Creating learning rate scheduler...")
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5 if not hasattr(self.config.training, 'scheduler_patience') 
                          else self.config.training.scheduler_patience,
                verbose=True
            )
        
        # Set up trainer using the modular architecture
        self.trainer = ModelTrainer(
            config=self.config,
            model=self.model,
            train_loader=self.dataloaders["train"],
            val_loader=self.dataloaders["val"] if "val" in self.dataloaders else None,
            optimizer=self.optimizer,
            scheduler=scheduler,
            logger=self.logger
        )
        
        # Set trainer state if resuming training
        if start_epoch > 0:
            self.trainer.current_epoch = start_epoch
            self.trainer.best_val_loss = best_val_loss
            
            if metrics_history:
                self.trainer.metrics_manager.metrics_history = metrics_history
                
            self.logger.info(f"Resuming training from epoch {start_epoch}")
            self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
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
        self.logger.info(f"Using curriculum learning: {curriculum}")
        
        # Train the model using the appropriate method based on curriculum flag
        if curriculum:
            if curriculum_params is None:
                curriculum_params = {}
                
            # Default curriculum parameters with strict stage progression
            stage_schedule = curriculum_params.get('stage_schedule', [
                (0, 50),  # Train first stage for 50 epochs
                (1, 50)   # Train second stage for 50 epochs
            ])
            
            learning_rates = curriculum_params.get('learning_rates', [
                self.config.training.learning_rate,
                self.config.training.learning_rate / 2
            ])
            
            # Always set stage_freezing to True for better GPU efficiency
            # This ensures only the current stage is active and unfrozen
            stage_freezing = curriculum_params.get('stage_freezing', [True] * len(stage_schedule))
            
            self.logger.info("Using curriculum learning approach with GPU optimization")
            self.logger.info(f"Stage schedule: {stage_schedule}")
            self.logger.info(f"Learning rates: {learning_rates}")
            self.logger.info(f"Stage freezing: {stage_freezing}")
            
            # Initialize curriculum manager if needed
            if not hasattr(self.trainer, 'curriculum_manager') or self.trainer.curriculum_manager is None:
                from ..models.curriculum_manager import CurriculumManager
                self.trainer.curriculum_manager = CurriculumManager(
                    config=self.trainer.config.training,
                    model=self.trainer.model,
                    logger=self.logger
                )
            
            # Configure curriculum with the updated parameters
            self.trainer.curriculum_manager.configure_curriculum({
                'stage_schedule': stage_schedule,
                'learning_rates': learning_rates,
                'stage_freezing': stage_freezing
            })
            
            # Use the curriculum training method
            training_results = self.trainer.train_with_curriculum(
                stage_schedule=stage_schedule,
                learning_rates=learning_rates,
                stage_freezing=stage_freezing
            )
        else:
            # Use the standard training method without curriculum
            self.logger.info("Using standard training approach (no curriculum)")
            training_results = self.trainer.train()
        
        self.logger.info("Model training completed")
        return training_results
    
    def evaluate(self):
        """
        Evaluate the trained model.
        """
        if self.trainer is None:
            self.logger.warning("Trainer not set up. Run setup_model() first.")
            return
        
        self.logger.info("Evaluating model...")
        
        # Evaluate on validation set using the new evaluate method
        val_metrics = self.trainer.evaluate(self.dataloaders.get("val"))
        
        # Evaluate on test set if available
        if "test" in self.dataloaders:
            self.logger.info("Evaluating on test set...")
            test_metrics = self.trainer.evaluate(self.dataloaders["test"])
            return {"val": val_metrics, "test": test_metrics}
        
        return {"val": val_metrics}
    
    def visualize(self):
        """
        Generate visualizations for the experiment.
        """
        if self.model is None or self.trainer is None:
            self.logger.warning("Model not set up. Run setup_model() first.")
            return
        
        self.logger.info("Generating visualizations...")
        
        # Use the visualization manager from the trainer to generate visualizations
        if hasattr(self.trainer, 'visualization_manager'):
            dataloader = self.dataloaders["val"] if "val" in self.dataloaders else self.dataloaders["train"]
            
            # Generate predictions visualizations
            self.trainer.visualization_manager.visualize_predictions(
                model=self.model,
                dataloader=dataloader,
                device=self.config.training.device,
                num_samples=8,  # Show more samples for better evaluation
                epoch=self.trainer.current_epoch
            )
            
            # Generate metrics plots
            self.trainer.visualization_manager.plot_metrics(
                metrics_history=self.trainer.metrics_manager.metrics_history,
                epoch=self.trainer.current_epoch,
                save=True
            )
            
            self.logger.info(f"Visualizations saved to {self.trainer.visualization_manager.visualizations_dir}")
        else:
            # Fallback to old method if needed
            from ..visualization.visualizer import generate_visualizations
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
        force_overwrite: bool = False,
        load_best: bool = True
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
            load_best: Whether to automatically load the best checkpoint if available
        """
        start_time = time.time()
        
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        try:
            # Process data if requested
            if process_data:
                self.process_data(
                    volume_dir=volume_dir,
                    segmentation_dir=segmentation_dir,
                    force_overwrite=force_overwrite
                )
            
            # Set up data pipeline
            self.setup_data_pipeline()
            
            # Set up model (with checkpoint loading control)
            self.setup_model(load_best=load_best)
            
            # If curriculum is True but no params provided, use the config's curriculum settings
            if curriculum and not curriculum_params and self.config.training.curriculum.enabled:
                curriculum_params = {
                    'stage_schedule': self.config.training.curriculum.stage_schedule,
                    'learning_rates': self.config.training.curriculum.learning_rates,
                    'stage_freezing': self.config.training.curriculum.stage_freezing,
                    'stage_overlap': self.config.training.curriculum.stage_overlap
                }
                
                # Log the curriculum parameters from config
                self.logger.info("Using curriculum parameters from config:")
                self.logger.info(f"  Stage Schedule: {curriculum_params['stage_schedule']}")
                self.logger.info(f"  Learning Rates: {curriculum_params['learning_rates']}")
                self.logger.info(f"  Stage Freezing: {curriculum_params['stage_freezing']}")
                if 'stage_overlap' in curriculum_params:
                    self.logger.info(f"  Stage Overlap: {curriculum_params['stage_overlap']}")
            
            # Train model with the appropriate curriculum settings
            training_results = self.train(
                curriculum=curriculum, 
                curriculum_params=curriculum_params
            )
            
            # Evaluate model
            evaluation_results = self.evaluate()
            
            # Generate visualizations
            self.visualize()
            
            total_time = time.time() - start_time
            self.logger.info(f"Experiment completed in {total_time/60:.2f} minutes")
            
            return {
                "model": self.model,
                "trainer": self.trainer,
                "datasets": self.datasets,
                "dataloaders": self.dataloaders,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "total_time_minutes": total_time/60
            }
            
        except Exception as e:
            self.logger.error(f"Error during experiment execution: {str(e)}")
            raise