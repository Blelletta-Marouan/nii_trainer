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
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            # Add channel dimension to target if missing
            target = target.unsqueeze(1)
            
        # Ensure target has the same dtype
        target = target.type_as(pred)
        
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
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            # Add channel dimension to target if missing
            target = target.unsqueeze(1)
        
        # Ensure target has the same dtype (use float for BCE)
        target = target.type_as(pred)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce
        return focal_loss.mean()

class MetricCalculator:
    """Calculates various metrics for reward computation."""
    def __init__(self):
        self.metric_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_cache_key(self, pred: torch.Tensor, target: torch.Tensor, threshold: float) -> str:
        # Create a unique key based on input tensors and threshold
        return f"{pred.shape}_{target.shape}_{threshold}_{pred.sum().item():.4f}_{target.sum().item():.4f}"
    
    def _safe_compute(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Safely compute all metrics at once to avoid redundant computations."""
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.type_as(pred)
        
        # Compute binary prediction once
        pred = (torch.sigmoid(pred) > threshold).float()
        
        # Basic metric components
        intersection = (pred * target).sum(dim=(2, 3))
        pred_sum = pred.sum(dim=(2, 3))
        target_sum = target.sum(dim=(2, 3))
        
        # True positives (needed for multiple metrics)
        tp = intersection
        
        # Compute metrics with safe division
        smooth = 1e-5  # Small constant to prevent division by zero
        
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        
        # Precision components
        fp = pred_sum - tp
        precision = (tp + smooth) / (tp + fp + smooth)
        
        # Recall components
        fn = target_sum - tp
        recall = (tp + smooth) / (tp + fn + smooth)
        
        # IoU components
        union = pred_sum + target_sum - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        # Average and validate all metrics
        metrics = {
            'dice': torch.nan_to_num(dice.mean(), nan=0.0, posinf=1.0, neginf=0.0),
            'precision': torch.nan_to_num(precision.mean(), nan=0.0, posinf=1.0, neginf=0.0),
            'recall': torch.nan_to_num(recall.mean(), nan=0.0, posinf=1.0, neginf=0.0),
            'iou': torch.nan_to_num(iou.mean(), nan=0.0, posinf=1.0, neginf=0.0)
        }
        
        return metrics

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Compute all metrics with caching."""
        cache_key = self._get_cache_key(pred, target, threshold)
        
        if cache_key in self.metric_cache:
            self.cache_hits += 1
            return self.metric_cache[cache_key]
        
        self.cache_misses += 1
        metrics = self._safe_compute(pred, target, threshold)
        
        # Store in cache
        self.metric_cache[cache_key] = metrics
        
        # Clear cache if it gets too large
        if len(self.metric_cache) > 1000:
            self.metric_cache.clear()
            
        return metrics

    # Individual metric accessors that use the cached computation
    def dice_coef(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.compute_metrics(pred, target, threshold)['dice']
        
    def precision(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.compute_metrics(pred, target, threshold)['precision']
        
    def recall(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.compute_metrics(pred, target, threshold)['recall']
        
    def jaccard(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.compute_metrics(pred, target, threshold)['iou']

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
        self.metrics = MetricCalculator()  # Uses our optimized MetricCalculator
        self.logger = logger or logging.getLogger(__name__)

    def compute_stage_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weight: float,
        threshold: float
    ) -> Dict[str, torch.Tensor]:
        """Compute loss and metrics for a single stage."""
        # Safety checks
        if not torch.isfinite(pred).all():
            self.logger.warning("Non-finite values in predictions, clipping...")
            pred = torch.clamp(pred, -100, 100)

        # Ensure target has same shape as predictions
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Basic losses with safe computation
        try:
            focal_loss = self.focal_loss(pred, target)
            dice_loss = self.dice_loss(pred, target)
            
            # Handle non-finite losses
            focal_loss = torch.nan_to_num(focal_loss, nan=0.5, posinf=1.0, neginf=0.0)
            dice_loss = torch.nan_to_num(dice_loss, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Safe base loss computation with clipping
            base_loss = (
                self.config.weight_bce * torch.clamp(focal_loss, 0.0, 10.0) +
                self.config.weight_dice * torch.clamp(dice_loss, 0.0, 10.0)
            )
            base_loss = torch.clamp(base_loss, 0.0, 10.0)
        except Exception as e:
            self.logger.error(f"Error computing losses: {e}")
            base_loss = torch.tensor(1.0, device=pred.device)
            focal_loss = torch.tensor(0.5, device=pred.device)
            dice_loss = torch.tensor(0.5, device=pred.device)

        # Use optimized metric computation with caching
        metrics = self.metrics.compute_metrics(pred, target, threshold)
        
        # Calculate reward safely
        try:
            reward_coef = getattr(self.config, 'reward_coef', 0.5)
            metric_score = (
                0.3 * metrics['dice'] +
                0.1 * metrics['precision'] +
                0.4 * metrics['recall'] +
                0.2 * metrics['iou']
            )
            reward = reward_coef * metric_score
            reward = torch.clamp(reward.detach(), 0.0, 0.9)
        except Exception as e:
            self.logger.warning(f"Error calculating reward: {e}")
            reward = torch.tensor(0.0, device=pred.device)

        # Apply reward and class weight
        loss_w_r = base_loss * (1.0 - reward)
        final_loss = class_weight * loss_w_r

        # Final safety check
        if not torch.isfinite(final_loss):
            self.logger.warning(f"Non-finite final loss detected: {final_loss}, using base loss")
            final_loss = class_weight * base_loss
            loss_w_r = base_loss

        return {
            'base_loss': base_loss.detach(),
            'loss_w_r': loss_w_r.detach(),
            'total_loss': final_loss,
            'dice': metrics['dice'].detach(),
            'precision': metrics['precision'].detach(),
            'recall': metrics['recall'].detach(),
            'jaccard': metrics['iou'].detach(),
            'reward': reward.detach()
        }

    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor,
        skip_stages: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing loss for all stages.
        
        Args:
            predictions: List of predictions from each stage (B, C, H, W)
            target: Ground truth target (B, C, H, W) or (B, H, W)
            skip_stages: Optional list of stage indices to skip (for curriculum learning)
            
        Returns:
            Dictionary containing total loss and stage-wise metrics
        """
        total_loss = 0
        metrics_dict = {}
        stage_metrics_list = []
        
        # Initialize skip_stages if not provided
        if skip_stages is None:
            skip_stages = []

        # Early optimization: If all stages are skipped, return zeros
        if len(skip_stages) == len(predictions):
            device = target.device
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'base_loss': torch.tensor(0.0, device=device),
                'loss_w_r': torch.tensor(0.0, device=device),
                'dice': torch.tensor(0.0, device=device),
                'precision': torch.tensor(0.0, device=device),
                'recall': torch.tensor(0.0, device=device),
                'jaccard': torch.tensor(0.0, device=device)
            }
        
        # Ensure target has same shape as predictions
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Get active stages
        active_stages = [i for i in range(len(predictions)) if i not in skip_stages]
        if not active_stages:
            return {'total_loss': torch.tensor(0.0, device=target.device)}
        
        # Track highest active stage for metric reporting
        current_active_stage = max(active_stages)
        
        # Process each stage
        for stage, pred in enumerate(predictions):
            # Skip stages not in curriculum
            if stage in skip_stages:
                continue
                
            # Get stage configuration
            weight = (self.config.stage_weights[stage] 
                     if self.config.stage_weights and stage < len(self.config.stage_weights) else 1.0)
            threshold = (self.config.threshold_per_class[stage] 
                        if self.config.threshold_per_class and stage < len(self.config.threshold_per_class) else 0.5)
            
            # Skip computation for placeholders (when using curriculum)
            if pred.shape[1] == 1 and pred.sum() == 0 and pred.shape[2:] != target.shape[2:]:
                continue
            
            try:
                # Compute metrics only for active stages
                stage_metrics = self.compute_stage_loss(pred, target, weight, threshold)
                
                # Add to total loss
                if torch.isfinite(stage_metrics['total_loss']):
                    total_loss += stage_metrics['total_loss']
                
                # Store stage-specific metrics
                for k, v in stage_metrics.items():
                    metrics_dict[f'stage{stage+1}_{k}'] = v
                
                # For the current active stage, store metrics without prefix for display
                if stage == current_active_stage:
                    for k, v in stage_metrics.items():
                        if k != 'total_loss':  # Don't overwrite total_loss
                            metrics_dict[k] = v
                
                stage_metrics['stage'] = stage
                stage_metrics_list.append(stage_metrics)
                
            except Exception as e:
                self.logger.error(f"Error computing metrics for stage {stage+1}: {e}")
                total_loss += torch.tensor(0.5, device=pred.device)
        
        # Add total loss to metrics
        metrics_dict['total_loss'] = total_loss
        metrics_dict['current_stage'] = current_active_stage + 1
        
        return metrics_dict