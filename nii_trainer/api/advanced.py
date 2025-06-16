"""
Advanced API for power users of NII-Trainer.
"""

import torch
import optuna
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path

from ..core.config import GlobalConfig
from ..models import create_model
from ..training import BaseTrainer
from ..evaluation import SegmentationEvaluator
from ..utils.logging import LoggerManager
from ..core.exceptions import NIITrainerError


class AdvancedTrainer:
    """
    Advanced trainer with full control over training process.
    """
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.callbacks = []
        self.hooks = []
    
    def build_model(self, model_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Build model with advanced configuration."""
        if model_config:
            self.config.model = self.config.model.__class__.from_dict(model_config)
        
        self.model = create_model(self.config.model)
        return self.model
    
    def setup_training(self, custom_components: Optional[Dict[str, Any]] = None) -> None:
        """Setup training with custom components."""
        if not self.model:
            raise NIITrainerError("Model must be built before setup")
        
        # Initialize trainer with custom components
        self.trainer = BaseTrainer(
            model=self.model,
            config=self.config,
            custom_components=custom_components or {}
        )
        
        # Setup evaluator
        self.evaluator = SegmentationEvaluator(self.config.evaluation)
    
    def add_callback(self, callback: Callable) -> None:
        """Add training callback."""
        self.callbacks.append(callback)
    
    def add_hook(self, hook: Callable) -> None:
        """Add training hook."""
        self.hooks.append(hook)
    
    def train_with_validation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        custom_metrics: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Train with custom validation logic."""
        if not self.trainer:
            raise NIITrainerError("Training must be setup before training")
        
        # Add custom metrics if provided
        if custom_metrics:
            for metric in custom_metrics:
                self.evaluator.add_metric(metric)
        
        # Execute callbacks
        for callback in self.callbacks:
            callback("on_train_start", self)
        
        # Train with hooks
        results = self.trainer.train_with_hooks(
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=self.hooks
        )
        
        # Execute callbacks
        for callback in self.callbacks:
            callback("on_train_end", self, results)
        
        return results


class ModelBuilder:
    """
    Advanced model builder with architecture search capabilities.
    """
    
    def __init__(self):
        self.components = {}
        self.search_space = {}
    
    def register_component(self, name: str, component_class: type) -> None:
        """Register custom component."""
        self.components[name] = component_class
    
    def define_search_space(self, search_space: Dict[str, Any]) -> None:
        """Define architecture search space."""
        self.search_space = search_space
    
    def build_from_config(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Build model from configuration."""
        return create_model(config)
    
    def build_from_search(self, trial: optuna.Trial) -> torch.nn.Module:
        """Build model from hyperparameter search trial."""
        # Sample hyperparameters from search space
        config = {}
        for param, space in self.search_space.items():
            if space['type'] == 'categorical':
                config[param] = trial.suggest_categorical(param, space['choices'])
            elif space['type'] == 'int':
                config[param] = trial.suggest_int(param, space['low'], space['high'])
            elif space['type'] == 'float':
                config[param] = trial.suggest_float(param, space['low'], space['high'])
        
        return self.build_from_config(config)
    
    def suggest_architecture(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal architecture based on constraints."""
        # Placeholder for architecture suggestion logic
        # This would use heuristics or learned models to suggest architectures
        return {
            "encoder": "resnet50",
            "decoder": "unet",
            "num_stages": 2,
            "use_attention": True
        }


class DataPipelineBuilder:
    """
    Advanced data pipeline builder.
    """
    
    def __init__(self):
        self.transforms = []
        self.datasets = {}
        self.loaders = {}
    
    def add_transform(self, transform: Callable, stage: str = "train") -> None:
        """Add transform to pipeline."""
        self.transforms.append((transform, stage))
    
    def build_dataset(
        self,
        data_path: str,
        dataset_type: str = "medical",
        custom_dataset_class: Optional[type] = None
    ) -> torch.utils.data.Dataset:
        """Build dataset with custom configurations."""
        if custom_dataset_class:
            dataset = custom_dataset_class(data_path)
        else:
            # Use default dataset based on type
            from ..data import MedicalImageDataset
            dataset = MedicalImageDataset(data_path)
        
        return dataset
    
    def build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        custom_sampler: Optional[torch.utils.data.Sampler] = None,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """Build dataloader with advanced options."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and custom_sampler is None,
            sampler=custom_sampler,
            **kwargs
        )
    
    def create_balanced_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        balance_strategy: str = "weighted"
    ) -> torch.utils.data.Sampler:
        """Create balanced sampler for imbalanced datasets."""
        # Implementation would depend on dataset structure
        if balance_strategy == "weighted":
            # Calculate class weights and create weighted sampler
            class_counts = self._calculate_class_counts(dataset)
            weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = weights[self._get_sample_labels(dataset)]
            return torch.utils.data.WeightedRandomSampler(
                sample_weights, len(sample_weights)
            )
        else:
            raise NotImplementedError(f"Balance strategy {balance_strategy} not implemented")
    
    def _calculate_class_counts(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Calculate class distribution in dataset."""
        # Placeholder - would need to be implemented based on dataset structure
        return [100, 200]  # Example counts
    
    def _get_sample_labels(self, dataset: torch.utils.data.Dataset) -> torch.Tensor:
        """Get labels for all samples in dataset."""
        # Placeholder - would need to be implemented based on dataset structure
        return torch.randint(0, 2, (len(dataset),))


class ExperimentManager:
    """
    Advanced experiment management with tracking and comparison.
    """
    
    def __init__(self, base_output_dir: str = "./experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = {}
        self.active_experiment = None
    
    def create_experiment(
        self,
        name: str,
        config: GlobalConfig,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """Create new experiment."""
        experiment_dir = self.base_output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_info = {
            "name": name,
            "config": config,
            "tags": tags or [],
            "description": description,
            "directory": experiment_dir,
            "status": "created",
            "results": {},
            "metrics_history": []
        }
        
        self.experiments[name] = experiment_info
        return name
    
    def set_active_experiment(self, name: str) -> None:
        """Set active experiment."""
        if name not in self.experiments:
            raise NIITrainerError(f"Experiment {name} not found")
        self.active_experiment = name
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for active experiment."""
        if not self.active_experiment:
            raise NIITrainerError("No active experiment")
        
        self.experiments[self.active_experiment]["metrics_history"].append({
            "step": step,
            "metrics": metrics
        })
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metric: str = "val_dice"
    ) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {}
        
        for name in experiment_names:
            if name not in self.experiments:
                continue
            
            exp = self.experiments[name]
            metrics_history = exp["metrics_history"]
            
            if metrics_history:
                metric_values = [m["metrics"].get(metric, 0) for m in metrics_history]
                comparison[name] = {
                    "best": max(metric_values) if metric_values else 0,
                    "final": metric_values[-1] if metric_values else 0,
                    "mean": np.mean(metric_values) if metric_values else 0,
                    "std": np.std(metric_values) if metric_values else 0
                }
        
        return comparison
    
    def get_best_experiment(self, metric: str = "val_dice") -> Optional[str]:
        """Get best experiment based on metric."""
        best_name = None
        best_value = float('-inf')
        
        for name, exp in self.experiments.items():
            metrics_history = exp["metrics_history"]
            if not metrics_history:
                continue
            
            metric_values = [m["metrics"].get(metric, 0) for m in metrics_history]
            if metric_values:
                best_exp_value = max(metric_values)
                if best_exp_value > best_value:
                    best_value = best_exp_value
                    best_name = name
        
        return best_name


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning with Optuna integration.
    """
    
    def __init__(
        self,
        objective_metric: str = "val_dice",
        direction: str = "maximize",
        n_trials: int = 100
    ):
        self.objective_metric = objective_metric
        self.direction = direction
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
    
    def define_search_space(self, search_space: Dict[str, Any]) -> None:
        """Define hyperparameter search space."""
        self.search_space = search_space
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        # Sample hyperparameters
        params = {}
        for param, space in self.search_space.items():
            if space['type'] == 'categorical':
                params[param] = trial.suggest_categorical(param, space['choices'])
            elif space['type'] == 'int':
                params[param] = trial.suggest_int(param, space['low'], space['high'])
            elif space['type'] == 'float':
                params[param] = trial.suggest_float(param, space['low'], space['high'])
        
        # Create and train model with sampled parameters
        # This would need to be implemented based on specific use case
        config = self._create_config_from_params(params)
        trainer = AdvancedTrainer(config)
        trainer.build_model()
        trainer.setup_training()
        
        # Quick training for hyperparameter search
        results = trainer.train_with_validation(
            train_loader=self._get_train_loader(),
            val_loader=self._get_val_loader()
        )
        
        return results.get(self.objective_metric, 0.0)
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials)
        }
    
    def _create_config_from_params(self, params: Dict[str, Any]) -> GlobalConfig:
        """Create configuration from parameters."""
        # Placeholder - would need to map parameters to config structure
        from ..core.config import create_default_config
        config = create_default_config()
        
        # Update config with parameters
        for key, value in params.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
        
        return config
    
    def _get_train_loader(self) -> torch.utils.data.DataLoader:
        """Get training dataloader - placeholder."""
        # This would need to be implemented based on specific dataset
        pass
    
    def _get_val_loader(self) -> torch.utils.data.DataLoader:
        """Get validation dataloader - placeholder."""
        # This would need to be implemented based on specific dataset
        pass