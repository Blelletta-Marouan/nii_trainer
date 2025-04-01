import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from ..configs.config import LossConfig

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce
        return focal_loss.mean()

class MetricCalculator:
    """Calculates various metrics for reward computation."""
    @staticmethod
    def dice_coef(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return dice.mean()
    
    @staticmethod
    def precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        pred = (torch.sigmoid(pred) > threshold).float()
        tp = (pred * target).sum(dim=(2, 3))
        fp = pred.sum(dim=(2, 3)) - tp
        precision = (tp + 1e-5) / (tp + fp + 1e-5)
        return precision.mean()
    
    @staticmethod
    def recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        pred = (torch.sigmoid(pred) > threshold).float()
        tp = (pred * target).sum(dim=(2, 3))
        fn = target.sum(dim=(2, 3)) - tp
        recall = (tp + 1e-5) / (tp + fn + 1e-5)
        return recall.mean()
    
    @staticmethod
    def jaccard(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        return iou.mean()

class CascadedLossWithReward(nn.Module):
    """
    Combined loss for cascaded segmentation with reward mechanism.
    Supports multiple stages and classes with configurable weights and thresholds.
    """
    def __init__(self, config: LossConfig, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=config.focal_gamma)
        self.metrics = MetricCalculator()
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Initializing CascadedLossWithReward")
        self.logger.debug(f"Loss config - BCE weight: {config.weight_bce}, Dice weight: {config.weight_dice}")
        self.logger.debug(f"Focal gamma: {config.focal_gamma}, Reward coefficient: {config.reward_coef}")
        if config.stage_weights:
            self.logger.debug(f"Stage weights: {config.stage_weights}")
        if config.class_weights:
            self.logger.debug(f"Class weights: {config.class_weights}")

    def compute_stage_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weight: float,
        threshold: float
    ) -> Dict[str, torch.Tensor]:
        """Compute loss and metrics for a single stage."""
        # Basic losses
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        base_loss = (
            self.config.weight_bce * focal_loss +
            self.config.weight_dice * dice_loss
        )
        
        # Compute metrics for reward
        with torch.no_grad():
            dice = self.metrics.dice_coef(pred, target, threshold)
            precision = self.metrics.precision(pred, target, threshold)
            recall = self.metrics.recall(pred, target, threshold)
            iou = self.metrics.jaccard(pred, target, threshold)
        
        # Compute reward
        reward = self.config.reward_coef * (
            0.2 * dice +
            0.2 * precision +
            0.3 * recall +
            0.3 * iou
        )
        
        # Final loss with class weight and reward
        final_loss = class_weight * (base_loss - reward)
        
        self.logger.debug(
            f"Stage metrics - Dice: {dice:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, IoU: {iou:.4f}, Reward: {reward:.4f}"
        )
        
        return {
            'loss': final_loss,
            'base_loss': base_loss,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'reward': reward
        }

    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing loss for all stages."""
        total_loss = 0
        metrics_dict = {}
        
        self.logger.debug(f"Computing loss for {len(predictions)} stages")
        
        for stage, (pred, stage_target) in enumerate(zip(predictions, target)):
            # Get stage configuration
            weight = (self.config.stage_weights[stage] 
                     if self.config.stage_weights else 1.0)
            threshold = (self.config.threshold_per_class[stage] 
                        if self.config.threshold_per_class else 0.5)
            
            self.logger.debug(
                f"Stage {stage+1} - Weight: {weight:.2f}, Threshold: {threshold:.2f}"
            )
            
            # Compute stage metrics
            stage_metrics = self.compute_stage_loss(
                pred, stage_target, weight, threshold
            )
            
            total_loss += stage_metrics['loss']
            
            # Store metrics
            for k, v in stage_metrics.items():
                metrics_dict[f'stage{stage+1}_{k}'] = v
                
        metrics_dict['total_loss'] = total_loss
        self.logger.debug(f"Total loss: {total_loss:.4f}")
        
        return metrics_dict