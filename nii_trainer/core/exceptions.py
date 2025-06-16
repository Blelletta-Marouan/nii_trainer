"""
Custom exceptions for the NII-Trainer package.
"""

class NIITrainerError(Exception):
    """Base exception for all NII-Trainer errors."""
    pass

class ConfigurationError(NIITrainerError):
    """Raised when there's an error in configuration."""
    pass

class ValidationError(NIITrainerError):
    """Raised when validation fails."""
    pass

class ModelError(NIITrainerError):
    """Raised when there's an error with model operations."""
    pass

class DataError(NIITrainerError):
    """Raised when there's an error with data operations."""
    pass

class TrainingError(NIITrainerError):
    """Raised when there's an error during training."""
    pass

class EvaluationError(NIITrainerError):
    """Raised when there's an error during evaluation."""
    pass