"""
NII-Trainer: Advanced Neural Network Training Framework for Medical Image Segmentation

A comprehensive PyTorch-based framework specifically designed for training neural networks
on medical imaging data, featuring cascaded architectures, advanced data processing,
and robust evaluation capabilities.

Author: NII-Trainer Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "NII-Trainer Development Team"
__email__ = "contact@nii-trainer.org"
__license__ = "MIT"

# Core components
from .core import (
    GlobalConfig,
    ModelConfig, 
    StageConfig,
    TrainingConfig,
    EvaluationConfig,
    ValidationError,
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    EvaluationError,
    ComponentRegistry,
    MODELS,
    ENCODERS,
    DECODERS,
    TRAINERS,
    EVALUATORS,
    TRANSFORMS,
    register_model,
    register_encoder,
    register_decoder,
    register_trainer,
    register_evaluator,
    register_transform
)

# Data processing
from .data import (
    read_medical_image,
    VolumeProcessor,
    SliceExtractor,
    MorphologicalProcessor,
    get_windowing_preset,
    get_training_transforms,
    get_validation_transforms,
    get_transform_preset
)

# Model architectures - now including base classes
from .models import (
    create_model,
    create_encoder,
    create_decoder,
    load_pretrained_model,
    get_model_summary,
    CascadedSegmentationModel,
    SimpleUNet,
    ProgressiveCascadeModel,
    # Base classes
    BaseModel,
    BaseEncoder,
    BaseDecoder,
    BaseCascadeModel,
    StageModule,
    AttentionFusion
)

# Training
from .training import (
    BaseTrainer,
    create_optimizer,
    create_scheduler,
    # Loss functions
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    IoULoss,
    BoundaryLoss,
    DiceBCELoss,
    DiceFocalLoss,
    MultiStageLoss,
    WeightedCombinedLoss,
    AdaptiveLoss,
    BalancedLoss,
    # Reward-based losses
    RewardBasedLoss,
    DualOutputRewardLoss,
    MultiMetricRewardLoss,
    ProgressiveRewardLoss,
    # Metric functions
    hard_dice_coef,
    hard_jaccard_index,
    hard_precision_metric,
    hard_recall_metric,
    create_loss
)

# Evaluation
from .evaluation import (
    SegmentationEvaluator,
    MetricsTracker,
    create_evaluator
)

# Utilities - now properly exposed
from .utils import (
    # Device management
    get_device,
    get_gpu_info,
    set_device,
    enable_mixed_precision,
    get_memory_usage,
    clear_gpu_cache,
    # Logging
    setup_logging,
    get_logger,
    log_system_info,
    log_model_info,
    log_experiment_config,
    create_tensorboard_logger,
    create_wandb_logger,
    LoggerManager,
    # I/O utilities
    save_checkpoint,
    load_checkpoint,
    save_predictions,
    load_predictions,
    create_directory,
    # Medical imaging utilities
    get_image_info,
    resample_image,
    crop_to_nonzero,
    pad_image,
    compute_bounding_box,
    apply_window_level,
    normalize_hu_values,
    # Memory optimization
    optimize_memory,
    get_memory_info,
    profile_memory,
    garbage_collect,
    # Debugging
    profile_function,
    debug_tensor,
    check_gradients,
    print_model_summary,
    benchmark_model
)

# Convenience imports for common use cases
from .models.cascaded import create_model as create_cascaded_model
from .data.readers import read_medical_image as load_medical_image
from .data.processors import VolumeProcessor as ImageProcessor


def get_version() -> str:
    """Get the current version of NII-Trainer."""
    return __version__


def get_supported_formats() -> list:
    """Get list of supported medical image formats."""
    return ['.nii', '.nii.gz', '.dcm', '.dicom', '.nrrd', '.nhdr', '.mha', '.mhd']


def get_available_models() -> list:
    """Get list of available model architectures."""
    return list(MODELS.registry.keys())


def get_available_encoders() -> list:
    """Get list of available encoder architectures."""
    return list(ENCODERS.registry.keys())


def get_available_decoders() -> list:
    """Get list of available decoder architectures."""
    return list(DECODERS.registry.keys())


def quick_start_config() -> GlobalConfig:
    """Create a quick start configuration for new users."""
    from .core.config import GlobalConfig, ModelConfig, StageConfig, TrainingConfig, EvaluationConfig
    
    # Create a simple 2-stage cascade configuration
    stage1 = StageConfig(
        stage_id=0,
        name="coarse_segmentation",
        encoder="resnet18",
        decoder="unet", 
        num_classes=2,
        depends_on=[],
        fusion_strategy="none"
    )
    
    stage2 = StageConfig(
        stage_id=1,
        name="fine_segmentation", 
        encoder="resnet50",
        decoder="unet",
        num_classes=2,
        depends_on=[0],
        fusion_strategy="concatenate"
    )
    
    model_config = ModelConfig(
        model_name="cascaded_segmentation",
        input_channels=1,
        num_stages=2,
        stages=[stage1, stage2],
        use_deep_supervision=True,
        use_attention=False,
        pretrained=True
    )
    
    training_config = TrainingConfig(
        max_epochs=100,
        batch_size=8,
        learning_rate=1e-3,
        optimizer="adam",
        scheduler="plateau",
        mixed_precision=True,
        early_stopping_patience=10,
        early_stopping_metric="dice_foreground",
        early_stopping_mode="max"
    )
    
    evaluation_config = EvaluationConfig(
        metrics=["dice", "iou", "precision", "recall"],
        threshold_optimization=True,
        compute_confidence_intervals=False,
        compute_hausdorff=False,
        compute_surface_distance=False
    )
    
    return GlobalConfig(
        model=model_config,
        training=training_config,
        evaluation=evaluation_config,
        device="auto",
        output_dir="./nii_trainer_output",
        seed=42,
        num_workers=4
    )


# Package-level utilities
__all__ = [
    # Version and info
    "__version__", "get_version", "get_supported_formats", 
    "get_available_models", "get_available_encoders", "get_available_decoders",
    "quick_start_config",
    
    # Core
    "GlobalConfig", "ModelConfig", "StageConfig", "TrainingConfig", "EvaluationConfig",
    "ValidationError", "ConfigurationError", "DataError", "ModelError", 
    "TrainingError", "EvaluationError",
    "register_model", "register_encoder", "register_decoder", 
    "register_trainer", "register_evaluator", "register_transform",
    
    # Data processing
    "read_medical_image", "load_medical_image", "VolumeProcessor", "ImageProcessor",
    "SliceExtractor", "MorphologicalProcessor", "get_windowing_preset",
    "get_training_transforms", "get_validation_transforms", "get_transform_preset",
    
    # Models and Base Classes
    "create_model", "create_cascaded_model", "create_encoder", "create_decoder",
    "load_pretrained_model", "get_model_summary", "CascadedSegmentationModel",
    "SimpleUNet", "ProgressiveCascadeModel",
    "BaseModel", "BaseEncoder", "BaseDecoder", "BaseCascadeModel", 
    "StageModule", "AttentionFusion",
    
    # Training and Loss Functions
    "BaseTrainer", "create_optimizer", "create_scheduler",
    # Basic losses
    "DiceLoss", "FocalLoss", "TverskyLoss", "IoULoss", "BoundaryLoss",
    # Composite losses  
    "DiceBCELoss", "DiceFocalLoss", "MultiStageLoss", "WeightedCombinedLoss",
    "AdaptiveLoss", "BalancedLoss",
    # Reward-based losses
    "RewardBasedLoss", "DualOutputRewardLoss", "MultiMetricRewardLoss", "ProgressiveRewardLoss",
    # Metric functions
    "hard_dice_coef", "hard_jaccard_index", "hard_precision_metric", "hard_recall_metric",
    # Loss factory
    "create_loss",
    
    # Evaluation
    "SegmentationEvaluator", "MetricsTracker", "create_evaluator",
    
    # Utilities
    "get_device", "get_gpu_info", "set_device", "enable_mixed_precision",
    "get_memory_usage", "clear_gpu_cache",
    "setup_logging", "get_logger", "log_system_info", "log_model_info",
    "log_experiment_config", "create_tensorboard_logger", "create_wandb_logger",
    "LoggerManager", "save_checkpoint", "load_checkpoint", "save_predictions",
    "load_predictions", "create_directory", "get_image_info", "resample_image",
    "crop_to_nonzero", "pad_image", "compute_bounding_box", "apply_window_level",
    "normalize_hu_values", "optimize_memory", "get_memory_info", "profile_memory",
    "garbage_collect", "profile_function", "debug_tensor", "check_gradients",
    "print_model_summary", "benchmark_model"
]

# Print welcome message on import
import sys
if 'pytest' not in sys.modules:  # Don't print during testing
    print(f"NII-Trainer v{__version__} - Advanced Medical Image Segmentation Framework")
    print("Visit https://github.com/nii-trainer/nii-trainer for documentation and examples")