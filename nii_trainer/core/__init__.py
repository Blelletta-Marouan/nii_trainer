"""
Core module for NII-Trainer.
"""

from .config import (
    ConfigBase,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    GlobalConfig,
    StageConfig,
    load_config,
    create_default_config
)

from .registry import (
    Registry,
    ComponentRegistry,
    MODELS,
    ENCODERS,
    DECODERS,
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
    DATASETS,
    TRAINERS,
    EVALUATORS,
    register_model,
    register_encoder,
    register_decoder,
    register_loss,
    register_optimizer,
    register_scheduler,
    register_transform,
    register_dataset,
    register_trainer,
    register_evaluator
)

from .exceptions import (
    NIITrainerError,
    ConfigurationError,
    ValidationError,
    ModelError,
    DataError,
    TrainingError,
    EvaluationError
)

__all__ = [
    # Configuration
    "ConfigBase",
    "DataConfig",
    "ModelConfig", 
    "TrainingConfig",
    "EvaluationConfig",
    "GlobalConfig",
    "StageConfig",
    "load_config",
    "create_default_config",
    
    # Registry
    "Registry",
    "ComponentRegistry",
    "MODELS", "ENCODERS", "DECODERS", "LOSSES",
    "OPTIMIZERS", "SCHEDULERS", "TRANSFORMS", "DATASETS",
    "TRAINERS", "EVALUATORS",
    "register_model", "register_encoder", "register_decoder",
    "register_loss", "register_optimizer", "register_scheduler",
    "register_transform", "register_dataset", "register_trainer",
    "register_evaluator",
    
    # Exceptions
    "NIITrainerError", "ConfigurationError", "ValidationError",
    "ModelError", "DataError", "TrainingError", "EvaluationError"
]