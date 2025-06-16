"""
Loss functions module for NII-Trainer.

This module provides comprehensive loss functions for medical image segmentation,
including basic losses, composite losses, and advanced reward-based losses.
"""

from .segmentation import (
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    IoULoss,
    BoundaryLoss
)

from .composite import (
    WeightedCombinedLoss,
    DiceBCELoss,
    DiceFocalLoss,
    MultiStageLoss,
    AdaptiveLoss,
    BalancedLoss
)

from .reward import (
    hard_dice_coef,
    hard_jaccard_index,
    hard_precision_metric,
    hard_recall_metric,
    RewardBasedLoss,
    DualOutputRewardLoss,
    MultiMetricRewardLoss,
    ProgressiveRewardLoss
)


def create_loss(loss_type: str, **kwargs):
    """
    Factory function to create loss functions by name.
    
    Args:
        loss_type: Name of the loss function
        **kwargs: Parameters for the loss function
        
    Returns:
        Loss function instance
    """
    loss_registry = {
        'dice': DiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'iou': IoULoss,
        'boundary': BoundaryLoss,
        'dice_bce': DiceBCELoss,
        'dice_focal': DiceFocalLoss,
        'multi_stage': MultiStageLoss,
        'reward': RewardBasedLoss,
        'dual_reward': DualOutputRewardLoss,
        'multi_metric_reward': MultiMetricRewardLoss,
        'progressive_reward': ProgressiveRewardLoss,
        'weighted_combined': WeightedCombinedLoss,
        'adaptive': AdaptiveLoss,
        'balanced': BalancedLoss
    }
    
    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_registry.keys())}")
    
    return loss_registry[loss_type](**kwargs)


__all__ = [
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