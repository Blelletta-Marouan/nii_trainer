"""
Configuration system for NII-Trainer.

This module provides hierarchical configuration management with YAML/JSON support,
runtime parameter validation, and type checking.
"""

import os
import yaml
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints
from dataclasses import dataclass, field, fields
from pathlib import Path

from .exceptions import ConfigurationError, ValidationError


class ConfigBase(ABC):
    """Abstract base class for all configuration objects."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with validation."""
        self._validate_and_set_attributes(**kwargs)
    
    def _validate_and_set_attributes(self, **kwargs):
        """Validate and set attributes with type checking."""
        # Get type hints for validation
        type_hints = get_type_hints(self.__class__)
        
        # Get default values from dataclass fields
        field_defaults = {}
        if hasattr(self.__class__, '__dataclass_fields__'):
            for field_name, field_info in self.__class__.__dataclass_fields__.items():
                if field_info.default != field_info.default_factory:
                    if field_info.default_factory != field_info.default_factory:
                        field_defaults[field_name] = field_info.default_factory()
                    else:
                        field_defaults[field_name] = field_info.default
        
        # Set attributes with validation
        for key, value in kwargs.items():
            if hasattr(self, key) or key in type_hints:
                # Type validation
                if key in type_hints:
                    expected_type = type_hints[key]
                    if not self._validate_type(value, expected_type):
                        raise ValidationError(
                            f"Invalid type for {key}: expected {expected_type}, got {type(value)}"
                        )
                setattr(self, key, value)
            else:
                raise ConfigurationError(f"Unknown configuration parameter: {key}")
    
    def _validate_type(self, value: Any, expected_type: Type) -> bool:
        """Validate that value matches expected type."""
        if value is None:
            return True  # Allow None for optional types
        
        # Handle Union types (like Optional)
        if hasattr(expected_type, '__origin__'):
            if expected_type.__origin__ is Union:
                return any(self._validate_type(value, t) for t in expected_type.__args__)
            elif expected_type.__origin__ in (list, List):
                if not isinstance(value, list):
                    return False
                if expected_type.__args__:
                    return all(self._validate_type(item, expected_type.__args__[0]) for item in value)
                return True
            elif expected_type.__origin__ in (dict, Dict):
                return isinstance(value, dict)
        
        return isinstance(value, expected_type)
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, ConfigBase):
                    result[key] = value.to_dict()
                elif isinstance(value, list) and value and isinstance(value[0], ConfigBase):
                    result[key] = [item.to_dict() for item in value]
                else:
                    result[key] = value
        return result
    
    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """Convert configuration to YAML string or save to file."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        return yaml_str
    
    def to_json(self, file_path: Optional[str] = None) -> str:
        """Convert configuration to JSON string or save to file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigBase':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'ConfigBase':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'ConfigBase':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def merge(self, other: 'ConfigBase') -> 'ConfigBase':
        """Merge with another configuration."""
        if not isinstance(other, self.__class__):
            raise ConfigurationError(f"Cannot merge {type(self)} with {type(other)}")
        
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
        
        deep_merge(merged_dict, other_dict)
        return self.__class__.from_dict(merged_dict)


@dataclass
class DataConfig(ConfigBase):
    """Configuration for data processing pipeline."""
    
    # Dataset paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: Optional[str] = None
    
    # Image processing
    image_size: List[int] = field(default_factory=lambda: [512, 512])
    slice_thickness: Optional[float] = None
    spacing: Optional[List[float]] = None
    
    # Windowing parameters
    window_level: Optional[float] = None
    window_width: Optional[float] = None
    hu_min: float = -1000
    hu_max: float = 1000
    
    # Preprocessing
    normalize: bool = True
    standardize: bool = True
    clip_values: bool = True
    
    # Augmentation
    augmentation: bool = True
    rotation_range: float = 15.0
    translation_range: float = 0.1
    scale_range: float = 0.1
    flip_probability: float = 0.5
    noise_std: float = 0.01
    
    # Batch processing
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    # Class balancing
    balance_classes: bool = True
    oversample_minority: bool = False
    undersample_majority: bool = False
    balance_strategy: str = "weighted"  # "weighted", "oversample", "undersample"
    
    def validate(self) -> None:
        """Validate data configuration."""
        if not self.train_data_path:
            raise ValidationError("train_data_path is required")
        
        if not os.path.exists(self.train_data_path):
            raise ValidationError(f"Training data path does not exist: {self.train_data_path}")
        
        if self.val_data_path and not os.path.exists(self.val_data_path):
            raise ValidationError(f"Validation data path does not exist: {self.val_data_path}")
        
        if len(self.image_size) not in [2, 3]:
            raise ValidationError("image_size must be 2D or 3D")
        
        if self.batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        
        if self.balance_strategy not in ["weighted", "oversample", "undersample"]:
            raise ValidationError("balance_strategy must be one of: weighted, oversample, undersample")


@dataclass
class StageConfig(ConfigBase):
    """Configuration for a single cascade stage."""
    
    stage_id: int = 0
    name: str = ""
    
    # Architecture
    encoder: str = "resnet50"
    decoder: str = "unet"
    num_classes: int = 2
    
    # Dependencies
    depends_on: List[int] = field(default_factory=list)
    fusion_strategy: str = "concatenate"  # "concatenate", "add", "attention"
    
    # Processing
    input_resolution: Optional[List[int]] = None
    output_resolution: Optional[List[int]] = None
    
    # Training specific
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    learning_rate: float = 1e-4
    
    def validate(self) -> None:
        """Validate stage configuration."""
        if self.stage_id < 0:
            raise ValidationError("stage_id must be non-negative")
        
        if self.num_classes < 2:
            raise ValidationError("num_classes must be at least 2")
        
        if self.fusion_strategy not in ["concatenate", "add", "attention"]:
            raise ValidationError("fusion_strategy must be one of: concatenate, add, attention")


@dataclass
class ModelConfig(ConfigBase):
    """Configuration for cascaded model architecture."""
    
    # Model identification
    model_name: str = "cascaded_segmentation"
    model_type: str = "cascade"
    
    # Cascade configuration
    num_stages: int = 2
    stages: List[StageConfig] = field(default_factory=list)
    
    # Global architecture settings
    backbone: str = "resnet50"
    pretrained: bool = True
    dropout_rate: float = 0.1
    
    # Input/Output
    input_channels: int = 1
    spatial_dims: int = 2  # 2D or 3D
    
    # Advanced features
    use_attention: bool = False
    use_deep_supervision: bool = False
    use_auxiliary_loss: bool = False
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.num_stages < 1:
            raise ValidationError("num_stages must be at least 1")
        
        if self.num_stages != len(self.stages):
            raise ValidationError("Number of stages must match length of stages list")
        
        if self.spatial_dims not in [2, 3]:
            raise ValidationError("spatial_dims must be 2 or 3")
        
        if self.input_channels < 1:
            raise ValidationError("input_channels must be positive")
        
        # Validate stage dependencies
        stage_ids = [stage.stage_id for stage in self.stages]
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in stage_ids:
                    raise ValidationError(f"Stage {stage.stage_id} depends on non-existent stage {dep}")
                if dep >= stage.stage_id:
                    raise ValidationError(f"Stage {stage.stage_id} cannot depend on later stage {dep}")


@dataclass
class TrainingConfig(ConfigBase):
    """Configuration for training process."""
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # "min" or "max"
    
    # Optimization
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9
    
    # Scheduling
    scheduler: str = "cosine"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 5
    
    # Loss configuration
    loss_function: str = "dice_bce"
    loss_weights: Optional[List[float]] = None
    class_weights: Optional[List[float]] = None
    
    # Stage-wise training
    training_strategy: str = "sequential"  # "sequential", "joint", "progressive", "alternating"
    stage_training_epochs: List[int] = field(default_factory=list)
    freeze_previous_stages: bool = True
    
    # Advanced training
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Checkpointing
    save_top_k: int = 3
    save_last: bool = True
    checkpoint_metric: str = "val_loss"
    checkpoint_mode: str = "min"
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.max_epochs <= 0:
            raise ValidationError("max_epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive")
        
        if self.early_stopping_mode not in ["min", "max"]:
            raise ValidationError("early_stopping_mode must be 'min' or 'max'")
        
        if self.training_strategy not in ["sequential", "joint", "progressive", "alternating"]:
            raise ValidationError("training_strategy must be one of: sequential, joint, progressive, alternating")
        
        if self.checkpoint_mode not in ["min", "max"]:
            raise ValidationError("checkpoint_mode must be 'min' or 'max'")


@dataclass
class EvaluationConfig(ConfigBase):
    """Configuration for evaluation process."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["dice", "iou", "precision", "recall"])
    compute_hausdorff: bool = False
    compute_surface_distance: bool = False
    
    # Thresholding
    threshold_optimization: bool = True
    threshold_metric: str = "dice"
    threshold_range: List[float] = field(default_factory=lambda: [0.1, 0.9])
    threshold_steps: int = 9
    
    # Post-processing
    remove_small_objects: bool = True
    min_object_size: int = 100
    fill_holes: bool = True
    
    # Visualization
    save_predictions: bool = True
    save_overlays: bool = True
    save_error_maps: bool = False
    
    # Statistical analysis
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Cross-validation
    cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "random", "group"
    
    def validate(self) -> None:
        """Validate evaluation configuration."""
        valid_metrics = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValidationError(f"Unknown metric: {metric}")
        
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValidationError("confidence_level must be between 0 and 1")
        
        if self.cv_folds < 2:
            raise ValidationError("cv_folds must be at least 2")
        
        if self.cv_strategy not in ["stratified", "random", "group"]:
            raise ValidationError("cv_strategy must be one of: stratified, random, group")


@dataclass
class GlobalConfig(ConfigBase):
    """Global configuration that combines all other configurations."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global settings
    experiment_name: str = "nii_trainer_experiment"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    
    # Environment
    use_deterministic: bool = True
    num_threads: int = 4
    
    def validate(self) -> None:
        """Validate global configuration."""
        # Validate sub-configurations
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        
        # Validate global settings
        if self.seed < 0:
            raise ValidationError("seed must be non-negative")
        
        if self.num_threads < 1:
            raise ValidationError("num_threads must be positive")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


def load_config(config_path: str) -> GlobalConfig:
    """Load configuration from file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yml', '.yaml']:
        return GlobalConfig.from_yaml(str(config_path))
    elif config_path.suffix.lower() == '.json':
        return GlobalConfig.from_json(str(config_path))
    else:
        raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")


def create_default_config() -> GlobalConfig:
    """Create a default configuration."""
    # Create default stage configurations
    stage1 = StageConfig(
        stage_id=0,
        name="organ_segmentation",
        encoder="resnet50",
        decoder="unet",
        num_classes=2,
        depends_on=[],
        learning_rate=1e-4
    )
    
    stage2 = StageConfig(
        stage_id=1,
        name="lesion_segmentation", 
        encoder="resnet50",
        decoder="unet",
        num_classes=2,
        depends_on=[0],
        fusion_strategy="concatenate",
        learning_rate=5e-5
    )
    
    # Create model config with stages
    model_config = ModelConfig(
        model_name="liver_tumor_cascade",
        num_stages=2,
        stages=[stage1, stage2],
        spatial_dims=2
    )
    
    return GlobalConfig(
        model=model_config,
        experiment_name="default_experiment"
    )