"""Configuration classes for the training pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Union
from pathlib import Path
import torch

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    save_dir: Path
    max_checkpoints: int = 5
    save_frequency: int = 1  # Save every N epochs
    save_best: bool = True
    save_last: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True

@dataclass
class VisualizationConfig:
    """Configuration for visualization management."""
    save_plots: bool = True
    save_predictions: bool = True
    plot_frequency: int = 1  # Plot every N epochs
    num_samples: int = 4  # Number of samples to visualize
    dpi: int = 100
    plot_size: Tuple[int, int] = (15, 10)
    
    # Visualization options
    show_class_colormap: bool = True
    overlay_alpha: float = 0.5
    show_metrics: bool = True
    show_confidence: bool = True
    
    # Logging options
    log_images: bool = True
    log_metrics: bool = True
    log_confusion_matrix: bool = True
    
    # Export options
    export_format: str = "png"  # png, jpg, pdf
    save_raw_predictions: bool = False  # Save raw prediction arrays
    
    # Layout options
    subplots_layout: Tuple[int, int] = (2, 2)  # rows, cols
    figure_title_fontsize: int = 16
    axis_label_fontsize: int = 12

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = False
    stage_schedule: List[Tuple[int, int]] = field(default_factory=list)  # [(stage_idx, num_epochs),...]
    learning_rates: List[float] = field(default_factory=list)  # Per-stage learning rates
    stage_freezing: List[bool] = field(default_factory=list)  # Whether to freeze previous stages
    stage_metrics: List[str] = field(default_factory=lambda: ['loss', 'dice'])  # Metrics to track per stage
    use_gradual_unfreezing: bool = False  # Whether to gradually unfreeze stages
    stage_overlap: int = 0  # Number of epochs to overlap between stages
    warm_up_epochs: int = 0  # Number of warm-up epochs per stage
    final_joint_training: bool = False  # Whether to do joint training of all stages at the end
    curriculum_metrics: Dict[str, float] = field(default_factory=lambda: {
        'loss_threshold': 0.3,  # Loss threshold to advance to next stage
        'dice_threshold': 0.8,  # Dice score threshold to advance
        'stability_epochs': 3  # Number of epochs metrics must be stable
    })

@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""
    metrics: List[str] = field(default_factory=lambda: ['loss', 'dice', 'iou', 'precision', 'recall'])
    track_per_class: bool = True
    log_to_file: bool = True
    log_frequency: int = 100  # Log every N batches

@dataclass
class ModelConfig:
    """Configuration for model architecture components."""
    encoder_type: str = "mobilenet_v2"  # mobilenet_v2, resnet18, resnet50, efficientnet
    in_channels: int = 1
    initial_features: int = 32
    num_layers: int = 5
    pretrained: bool = True

@dataclass
class StageConfig:
    """Configuration for a single stage in the cascaded model."""
    input_classes: List[str]  # Classes to consider as input
    target_class: str        # Class to segment in this stage
    encoder_type: str = "mobilenet_v2"  # Encoder backbone for this stage
    num_layers: int = 5  # Number of layers for both encoder and decoder
    skip_connections: bool = True
    dropout_rate: float = 0.3
    threshold: float = 0.5   # Prediction threshold for this stage
    is_binary: bool = False  # Whether this stage performs binary segmentation

@dataclass
class CascadeConfig:
    """Configuration for the entire cascade of models."""
    stages: List[StageConfig]
    in_channels: int = 1
    initial_features: int = 32  # Initial number of features
    feature_growth: float = 2.0  # Feature multiplication factor between stages
    pretrained: bool = True

@dataclass
class DataConfig:
    base_dir: str
    output_dir: str
    classes: List[str]  # All possible classes in order of segmentation
    class_map: Dict[str, int]  # Mapping of class names to indices
    img_size: Tuple[int, int] = (512, 512)
    batch_size: int = 16
    num_workers: int = 0  # -1 means use all available CPU cores
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    slice_step: int = 1
    skip_empty: bool = True
    balance_dataset: bool = True
    
    # Advanced preprocessing options
    normalization: Dict[str, Any] = field(default_factory=lambda: {
        "method": "min_max",  # Options: min_max, z_score, percentile
        "params": {
            "min_val": -1000,
            "max_val": 1000,
            "percentile_range": (1, 99)
        }
    })
    
    # Window parameters per modality/class
    window_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "default": {"width": 180, "level": 50},
        "liver": {"width": 150, "level": 30},
        "tumor": {"width": 200, "level": 70}
    })
    
    # Comprehensive augmentation settings
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "spatial": {
            "rotation_range": (-10, 10),
            "scale_range": (0.9, 1.1),
            "flip_probability": 0.5,
            "elastic_deformation": {"sigma": 10, "points": 3}
        },
        "intensity": {
            "brightness_range": (-0.2, 0.2),
            "contrast_range": (0.8, 1.2),
            "gamma_range": (0.8, 1.2),
            "noise": {
                "gaussian": {"mean": 0, "std": 0.02},
                "poisson_noise_lambda": 1.0
            }
        },
        "mixing": {
            "mixup_alpha": 0.2,
            "cutmix_prob": 0.3
        }
    })
    
    # Cache settings
    caching: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_cache_size_gb": 32,
        "cache_type": "memory",  # memory or disk
        "prefetch_factor": 2
    })

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Core training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-8
    epochs: int = 60
    batch_accumulation: int = 1
    device: str = None
    mixed_precision: bool = True
    gradient_clip_value: Optional[float] = None

    # Optimizer settings
    optimizer_type: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False
    })

    # Learning rate scheduler
    scheduler_type: str = "plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "min",
        "factor": 0.1,
        "patience": 3,
        "verbose": True,
        "min_lr": 1e-6
    })

    # Early stopping
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "patience": 11,
        "min_delta": 0.0001,
        "monitor": "val_loss",
        "mode": "min"
    })

    # Data loading
    dataloader_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True
    })

    # Gradient accumulation
    accumulation_params: Dict[str, Any] = field(default_factory=lambda: {
        "gradient_clip_norm": None,
        "skip_nan": True,
        "sync_grad": True
    })

    # Manager configurations
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(save_dir=Path("checkpoints")))
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class LossConfig:
    """Loss configuration with per-stage settings."""
    weight_bce: float = 0.5
    weight_dice: float = 0.5
    focal_gamma: float = 2.0
    reward_coef: float = 0.5  
    stage_weights: Optional[List[float]] = None  # Weight for each stage's loss
    class_weights: Optional[Dict[str, float]] = None  # Weight for each class
    threshold_per_class: Optional[List[float]] = None  # Threshold for each class
    
    def __post_init__(self):
        # Set default thresholds if not provided
        if self.threshold_per_class is None:
            self.threshold_per_class = [0.5] * 3  # Default threshold for each stage

@dataclass
class TrainerConfig:
    """Main configuration class combining all configs."""
    data: DataConfig
    cascade: CascadeConfig
    training: TrainingConfig
    loss: LossConfig
    experiment_name: str
    save_dir: Union[str, Path] = "./experiments"

    def __post_init__(self):
        # Convert save_dir to Path
        self.save_dir = Path(self.save_dir)
        
        # Update checkpoint save directory to include experiment name
        self.training.checkpoint.save_dir = self.save_dir / self.experiment_name / "checkpoints"
        
        # Validate configurations
        self._validate_cascade_config()
        self._validate_curriculum_config()
        
    def _validate_cascade_config(self):
        """Validate cascade configuration."""
        # Check classes in stages exist in data.classes
        all_stage_classes = set()
        for stage in self.cascade.stages:
            all_stage_classes.update(stage.input_classes)
            all_stage_classes.add(stage.target_class)
            
        unknown_classes = all_stage_classes - set(self.data.classes)
        if unknown_classes:
            raise ValueError(f"Unknown classes in stage configs: {unknown_classes}")
            
        # Validate stage progression
        processed_classes = set()
        for i, stage in enumerate(self.cascade.stages):
            unprocessed = set(stage.input_classes) - processed_classes - {stage.target_class}
            if unprocessed and i > 0:
                raise ValueError(f"Stage {i} requires unprocessed classes: {unprocessed}")
            processed_classes.add(stage.target_class)
            
    def _validate_curriculum_config(self):
        """Validate curriculum learning configuration."""
        if not self.training.curriculum.enabled:
            return
            
        curriculum = self.training.curriculum
        num_stages = len(self.cascade.stages)
        
        # Validate stage schedule
        if not curriculum.stage_schedule:
            curriculum.stage_schedule = [(i, self.training.epochs // num_stages) 
                                      for i in range(num_stages)]
        
        # Validate learning rates
        if not curriculum.learning_rates:
            curriculum.learning_rates = [self.training.learning_rate] * num_stages
            
        # Validate stage freezing
        if not curriculum.stage_freezing:
            curriculum.stage_freezing = [False] * num_stages
            
        # Validate lengths match number of stages
        if len(curriculum.stage_schedule) != num_stages:
            raise ValueError("Curriculum stage schedule must match number of model stages")
        if len(curriculum.learning_rates) != num_stages:
            raise ValueError("Curriculum learning rates must match number of model stages")
        if len(curriculum.stage_freezing) != num_stages:
            raise ValueError("Curriculum stage freezing must match number of model stages")