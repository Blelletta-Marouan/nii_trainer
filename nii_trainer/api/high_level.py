"""
High-level API for easy usage of NII-Trainer.
"""

import torch
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from ..core.config import GlobalConfig, create_default_config
from ..models import create_model
from ..training import BaseTrainer
from ..evaluation import SegmentationEvaluator
from ..data import read_medical_image
from ..utils.logging import LoggerManager
from ..utils.io import save_checkpoint, load_checkpoint
from ..core.exceptions import NIITrainerError


class NIITrainer:
    """
    Simple, high-level interface for medical image segmentation.
    
    This class provides a simplified API that allows users to get started
    with cascaded medical image segmentation in just a few lines of code.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any], GlobalConfig]] = None,
        experiment_name: str = "nii_experiment",
        output_dir: str = "./outputs"
    ):
        """
        Initialize NIITrainer.
        
        Args:
            config: Configuration file path, dict, or GlobalConfig object
            experiment_name: Name for this experiment
            output_dir: Directory to save outputs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create configuration
        if config is None:
            self.config = create_default_config()
        elif isinstance(config, str):
            from ..core.config import load_config
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = GlobalConfig.from_dict(config)
        elif isinstance(config, GlobalConfig):
            self.config = config
        else:
            raise NIITrainerError(f"Invalid config type: {type(config)}")
        
        # Update config with experiment settings
        self.config.experiment_name = experiment_name
        self.config.output_dir = str(self.output_dir)
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.logger_manager = None
    
    def setup(self, train_data_path: str, val_data_path: Optional[str] = None) -> None:
        """
        Setup the trainer with data paths.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data (optional)
        """
        # Update data configuration
        self.config.data.train_data_path = train_data_path
        if val_data_path:
            self.config.data.val_data_path = val_data_path
        
        # Validate configuration
        self.config.validate()
        
        # Initialize logger
        self.logger_manager = LoggerManager(
            experiment_name=self.experiment_name,
            log_dir=str(self.output_dir / "logs"),
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Create model
        self.model = create_model(self.config.model)
        
        # Initialize trainer
        self.trainer = BaseTrainer(
            model=self.model,
            config=self.config,
            logger=self.logger_manager.logger
        )
        
        # Initialize evaluator
        self.evaluator = SegmentationEvaluator(self.config.evaluation)
    
    def train(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train (overrides config)
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        if self.trainer is None:
            raise NIITrainerError("Must call setup() before training")
        
        # Override epochs if specified
        if epochs is not None:
            self.config.training.max_epochs = epochs
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Log system and model info
        from ..utils.logging import log_system_info, log_model_info
        log_system_info(self.logger_manager.logger)
        log_model_info(self.model, self.logger_manager.logger)
        
        # Train the model
        results = self.trainer.train()
        
        return results
    
    def evaluate(
        self,
        test_data_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            test_data_path: Path to test data (uses validation data if not provided)
            checkpoint_path: Path to model checkpoint to load
            
        Returns:
            Evaluation results dictionary
        """
        if self.evaluator is None:
            raise NIITrainerError("Must call setup() before evaluation")
        
        # Load checkpoint if specified
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        if self.model is None:
            raise NIITrainerError("No model available for evaluation")
        
        # Update data path if provided
        if test_data_path:
            self.config.data.test_data_path = test_data_path
        
        # Evaluate the model
        results = self.evaluator.evaluate(self.model)
        
        return results
    
    def predict(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save prediction (optional)
            checkpoint_path: Path to model checkpoint to load
            
        Returns:
            Prediction tensor
        """
        if self.model is None:
            raise NIITrainerError("No model available for prediction")
        
        # Load checkpoint if specified
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Load and preprocess image
        image = read_medical_image(image_path)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(image)
        
        # Save prediction if output path provided
        if output_path:
            from ..utils.io import save_predictions
            save_predictions(
                {"prediction": prediction, "input_path": image_path},
                output_path
            )
        
        return prediction
    
    def save_checkpoint(
        self,
        filepath: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint."""
        if filepath is None:
            filepath = self.output_dir / f"{self.experiment_name}_checkpoint.pth"
        
        if self.trainer is None:
            raise NIITrainerError("No trainer available")
        
        save_checkpoint(
            model=self.model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.lr_scheduler,
            epoch=self.trainer.current_epoch,
            loss=self.trainer.best_loss,
            metrics=self.trainer.best_metrics,
            filepath=filepath,
            metadata=metadata
        )
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        if self.model is None:
            raise NIITrainerError("No model available")
        
        checkpoint_data = load_checkpoint(
            filepath=filepath,
            model=self.model,
            optimizer=self.trainer.optimizer if self.trainer else None,
            scheduler=self.trainer.lr_scheduler if self.trainer else None
        )
        
        if self.trainer:
            self.trainer.current_epoch = checkpoint_data.get('epoch', 0)
            self.trainer.best_loss = checkpoint_data.get('loss', float('inf'))
            self.trainer.best_metrics = checkpoint_data.get('metrics', {})
    
    def export_model(
        self,
        format: str = "onnx",
        filepath: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None
    ) -> str:
        """
        Export model for deployment.
        
        Args:
            format: Export format ('onnx', 'torchscript')
            filepath: Output file path
            input_shape: Input tensor shape for tracing
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise NIITrainerError("No model available for export")
        
        if filepath is None:
            filepath = self.output_dir / f"{self.experiment_name}_model.{format}"
        
        if input_shape is None:
            # Use default shape from config
            if hasattr(self.config.data, 'image_size'):
                input_shape = (1, *self.config.data.image_size)
            else:
                input_shape = (1, 512, 512)  # Default
        
        self.model.eval()
        
        if format.lower() == "onnx":
            dummy_input = torch.randn(input_shape)
            torch.onnx.export(
                self.model,
                dummy_input,
                str(filepath),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        elif format.lower() == "torchscript":
            dummy_input = torch.randn(input_shape)
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(str(filepath))
        else:
            raise NIITrainerError(f"Unsupported export format: {format}")
        
        return str(filepath)
    
    def close(self) -> None:
        """Clean up resources."""
        if self.logger_manager:
            self.logger_manager.close()


def quick_train(
    train_data_path: str,
    val_data_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    epochs: int = 50,
    experiment_name: str = "quick_train",
    output_dir: str = "./outputs"
) -> NIITrainer:
    """
    Quick training function for rapid prototyping.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        model_config: Model configuration dictionary
        epochs: Number of epochs to train
        experiment_name: Name for this experiment
        output_dir: Directory to save outputs
        
    Returns:
        Trained NIITrainer instance
    """
    # Create trainer
    trainer = NIITrainer(
        config=model_config,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    # Setup and train
    trainer.setup(train_data_path, val_data_path)
    trainer.train(epochs=epochs)
    
    return trainer


def quick_evaluate(
    test_data_path: str,
    checkpoint_path: str,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    experiment_name: str = "quick_eval"
) -> Dict[str, Any]:
    """
    Quick evaluation function.
    
    Args:
        test_data_path: Path to test data
        checkpoint_path: Path to model checkpoint
        config: Configuration file path or dictionary
        experiment_name: Name for this experiment
        
    Returns:
        Evaluation results dictionary
    """
    # Create evaluator
    evaluator = NIITrainer(
        config=config,
        experiment_name=experiment_name
    )
    
    # Setup and evaluate
    evaluator.setup(test_data_path)
    results = evaluator.evaluate(
        test_data_path=test_data_path,
        checkpoint_path=checkpoint_path
    )
    
    return results


def create_model_from_config(config_path: str) -> torch.nn.Module:
    """Create model from configuration file."""
    from ..core.config import load_config
    config = load_config(config_path)
    return create_model(config.model)


def load_pretrained_model(
    model_name: str,
    checkpoint_path: Optional[str] = None
) -> torch.nn.Module:
    """Load a pretrained model."""
    # This would integrate with model zoo in the future
    if checkpoint_path:
        model = create_model({"model_name": model_name})
        load_checkpoint(checkpoint_path, model=model)
        return model
    else:
        # Load from model zoo (placeholder for future implementation)
        raise NIITrainerError(f"Model zoo not yet implemented for {model_name}")


def export_model(
    model: torch.nn.Module,
    output_path: str,
    format: str = "onnx",
    input_shape: Tuple[int, ...] = (1, 1, 512, 512)
) -> None:
    """Export model for deployment."""
    model.eval()
    
    if format.lower() == "onnx":
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
    elif format.lower() == "torchscript":
        dummy_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
    else:
        raise NIITrainerError(f"Unsupported export format: {format}")