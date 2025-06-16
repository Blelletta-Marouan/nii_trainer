"""
Training module for NII-Trainer.
"""

from .base import (
    BaseTrainer,
    BaseMetric,
    AverageMeter,
    create_optimizer,
    create_scheduler
)

from .losses import (
    # Segmentation losses
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    IoULoss,
    BoundaryLoss,
    
    # Composite losses
    WeightedCombinedLoss,
    DiceBCELoss,
    DiceFocalLoss,
    MultiStageLoss,
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
    
    # Factory function
    create_loss
)

__all__ = [
    # Base training classes
    "BaseTrainer",
    "BaseMetric", 
    "AverageMeter",
    "create_optimizer",
    "create_scheduler",
    
    # Segmentation losses
    "DiceLoss",
    "FocalLoss",
    "TverskyLoss",
    "IoULoss",
    "BoundaryLoss",
    
    # Composite losses
    "WeightedCombinedLoss",
    "DiceBCELoss", 
    "DiceFocalLoss",
    "MultiStageLoss",
    "AdaptiveLoss",
    "BalancedLoss",
    
    # Reward-based losses
    "RewardBasedLoss",
    "DualOutputRewardLoss",
    "MultiMetricRewardLoss",
    "ProgressiveRewardLoss",
    
    # Metric functions
    "hard_dice_coef",
    "hard_jaccard_index", 
    "hard_precision_metric",
    "hard_recall_metric",
    
    # Factory function
    "create_loss"
]