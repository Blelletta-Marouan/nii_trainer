"""
Evaluation module for NII-Trainer.
"""

from .evaluator import (
    BaseEvaluator,
    SegmentationEvaluator,
    MetricsTracker,
    create_evaluator
)

__all__ = [
    "BaseEvaluator",
    "SegmentationEvaluator", 
    "MetricsTracker",
    "create_evaluator"
]