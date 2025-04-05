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
    @staticmethod
    def dice_coef(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.type_as(pred)
            
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return dice.mean()
    
    @staticmethod
    def precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.type_as(pred)
            
        pred = (torch.sigmoid(pred) > threshold).float()
        tp = (pred * target).sum(dim=(2, 3))
        fp = pred.sum(dim=(2, 3)) - tp
        precision = (tp + 1e-5) / (tp + fp + 1e-5)
        return precision.mean()
    
    @staticmethod
    def recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.type_as(pred)
            
        pred = (torch.sigmoid(pred) > threshold).float()
        tp = (pred * target).sum(dim=(2, 3))
        fn = target.sum(dim=(2, 3)) - tp
        recall = (tp + 1e-5) / (tp + fn + 1e-5)
        return recall.mean()
    
    @staticmethod
    def jaccard(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # Ensure target has the same shape as pred
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.type_as(pred)
            
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
        """
        Compute loss and metrics for a single stage.
        
        Args:
            pred: Model predictions (B, C, H, W)
            target: Ground truth targets (B, C, H, W)
            class_weight: Weight for this class/stage
            threshold: Threshold for binary predictions
        """
        # Ensure target has same shape as predictions
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Basic losses
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        # Check for valid loss values to prevent numerical instability
        if not torch.isfinite(focal_loss):
            self.logger.warning(f"Non-finite focal loss detected: {focal_loss}, using default value")
            focal_loss = torch.tensor(0.5, device=pred.device)
            
        if not torch.isfinite(dice_loss):
            self.logger.warning(f"Non-finite dice loss detected: {dice_loss}, using default value")
            dice_loss = torch.tensor(0.5, device=pred.device)
        
        # Compute base loss with clipping - this is without reward
        base_loss = (
            self.config.weight_bce * torch.clamp(focal_loss, 0.0, 10.0) +
            self.config.weight_dice * torch.clamp(dice_loss, 0.0, 10.0)
        )
        base_loss = torch.clamp(base_loss, 0.0, 10.0)
        
        # Compute metrics for reward with safe computation
        with torch.no_grad():
            try:
                dice = self.metrics.dice_coef(pred, target, threshold)
                precision = self.metrics.precision(pred, target, threshold)
                recall = self.metrics.recall(pred, target, threshold)
                iou = self.metrics.jaccard(pred, target, threshold)
                
                # Validate metric values
                dice = 0.0 if not torch.isfinite(dice) else torch.clamp(dice, 0.0, 1.0)
                precision = 0.0 if not torch.isfinite(precision) else torch.clamp(precision, 0.0, 1.0)
                recall = 0.0 if not torch.isfinite(recall) else torch.clamp(recall, 0.0, 1.0)
                iou = 0.0 if not torch.isfinite(iou) else torch.clamp(iou, 0.0, 1.0)
            except Exception as e:
                self.logger.warning(f"Error calculating metrics: {e}")
                dice = torch.tensor(0.0, device=pred.device)
                precision = torch.tensor(0.0, device=pred.device)
                recall = torch.tensor(0.0, device=pred.device)
                iou = torch.tensor(0.0, device=pred.device)
        
        # FIXED REWARD CALCULATION: Calculate reward based on metrics and ensure it REDUCES the loss
        try:
            # Get reward coefficient from config (default to 0.5 if not specified)
            reward_coef = getattr(self.config, 'reward_coef', 0.5)
            
            # Calculate weighted metric score (higher is better)
            # Fine-tuned weights to prioritize recall and dice for medical imaging
            metric_score = (
                0.3 * dice +        # Overall segmentation quality
                0.1 * precision +   # Lower importance to avoid over-penalizing false positives  
                0.4 * recall +      # Critical for medical applications (don't miss regions)
                0.2 * iou           # Spatial overlap accuracy
            )
            
            # Calculate reward factor (between 0 and reward_coef)
            # Higher metrics mean higher reward
            reward = reward_coef * metric_score
            
            # Clamp to valid range, detach to prevent gradient tracking
            reward = torch.clamp(reward.detach(), 0.0, 0.9)
        except Exception as e:
            self.logger.warning(f"Error calculating reward: {e}")
            reward = torch.tensor(0.0, device=pred.device)
        
        # *** CRITICAL FIX: Apply reward by REDUCING the base loss ***
        # Higher reward = lower loss = better encouragement for good predictions
        loss_w_r = base_loss * (1.0 - reward)  # Multiplicative reduction
        
        # Debug logging (only for a small percentage of batches to avoid log flooding)
        if torch.rand(1).item() < 0.01:  # Log for ~1% of batches
            self.logger.debug(
                f"Loss: base={base_loss.item():.4f}, with_reward={loss_w_r.item():.4f}, "
                f"reduction={base_loss.item()-loss_w_r.item():.4f} "
                f"({(1.0-loss_w_r.item()/base_loss.item())*100:.1f}%), "
                f"metrics=[d={dice.item():.2f}, p={precision.item():.2f}, r={recall.item():.2f}]"
            )
        
        # *** CRITICAL FIX: Use loss_w_r as the final loss for backpropagation ***
        # This ensures we evaluate and train the model using the reward-adjusted loss
        final_loss = class_weight * loss_w_r
        
        # Make sure all values are valid
        if not torch.isfinite(final_loss):
            self.logger.warning(f"Non-finite final loss detected: {final_loss}, using base loss")
            final_loss = class_weight * base_loss
        
        return {
            'base_loss': base_loss.detach(),  # Original loss without reward
            'loss_w_r': loss_w_r.detach(),   # Loss with reward (should be <= base_loss)
            'total_loss': final_loss,        # Final loss with class weight (for backprop) - this is loss_w_r with class weight
            'dice': dice.detach(),
            'precision': precision.detach(),
            'recall': recall.detach(),
            'jaccard': iou.detach(),
            'reward': reward.detach()       # Raw reward value
        }

    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing loss for all stages.
        
        Args:
            predictions: List of predictions from each stage (B, C, H, W)
            target: Ground truth target (B, C, H, W) or (B, H, W)
            
        Returns:
            Dictionary containing total loss and stage-wise metrics
        """
        total_loss = 0
        metrics_dict = {}
        stage_metrics_list = []
        
        self.logger.debug(f"Computing loss for {len(predictions)} stages")
        
        # Ensure target has same shape as predictions
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Validate input predictions
        if not predictions or not all(torch.isfinite(p).all() for p in predictions):
            self.logger.warning("Invalid predictions detected, using default loss")
            return {'total_loss': torch.tensor(1.0, device=target.device)}
        
        # Active stages for curriculum learning (set by trainer)
        active_stages = getattr(self, 'active_stages', list(range(len(predictions))))
        
        # Track metrics from the most advanced active stage for reporting
        # (ensures metrics reflect current training focus, not all stages)
        current_active_stage = max(active_stages) if active_stages else 0
        
        # Process each stage
        for stage, pred in enumerate(predictions):
            # Get stage configuration
            weight = (self.config.stage_weights[stage] 
                     if self.config.stage_weights and stage < len(self.config.stage_weights) else 1.0)
            threshold = (self.config.threshold_per_class[stage] 
                        if self.config.threshold_per_class and stage < len(self.config.threshold_per_class) else 0.5)
            
            # Compute stage metrics with error handling
            try:
                stage_metrics = self.compute_stage_loss(
                    pred, target, weight, threshold
                )
                
                # Add to total loss if valid
                if torch.isfinite(stage_metrics['total_loss']):
                    total_loss += stage_metrics['total_loss']
                else:
                    self.logger.warning(f"Stage {stage+1} produced invalid loss, using default")
                    total_loss += torch.tensor(0.5, device=pred.device)
                
                # Store metrics with stage prefix
                for k, v in stage_metrics.items():
                    metrics_dict[f'stage{stage+1}_{k}'] = v
                
                # Ensure all required metrics are present with defaults if missing
                required_metrics = ['base_loss', 'total_loss', 'dice', 'precision', 'recall', 'jaccard', 'reward']
                metrics_to_store = {}
                
                for key in required_metrics:
                    if key in stage_metrics:
                        metrics_to_store[key] = stage_metrics[key]
                    else:
                        # Provide default values for missing metrics
                        metrics_to_store[key] = torch.tensor(0.0, device=pred.device)
                        self.logger.warning(f"Missing '{key}' in stage {stage+1} metrics, using default value 0.0")
                
                # Store stage metrics for aggregation
                metrics_to_store['stage'] = stage
                stage_metrics_list.append(metrics_to_store)
                
            except Exception as e:
                self.logger.error(f"Error in stage {stage+1}: {e}")
                # Add a default loss contribution
                total_loss += torch.tensor(0.5, device=pred.device if torch.is_tensor(pred) else "cpu")
                
                # Add default metrics for this stage
                default_device = pred.device if torch.is_tensor(pred) else "cpu"
                default_metrics = {
                    'stage': stage,
                    'base_loss': torch.tensor(0.5, device=default_device),
                    'total_loss': torch.tensor(0.5, device=default_device),
                    'dice': torch.tensor(0.0, device=default_device),
                    'precision': torch.tensor(0.0, device=default_device),
                    'recall': torch.tensor(0.0, device=default_device),
                    'jaccard': torch.tensor(0.0, device=default_device),
                    'reward': torch.tensor(0.0, device=default_device)
                }
                
                # Add default metrics with stage prefix
                for k, v in default_metrics.items():
                    if k != 'stage':
                        metrics_dict[f'stage{stage+1}_{k}'] = v
                
                # Add to stage metrics list
                stage_metrics_list.append(default_metrics)
        
        # Add total loss to metrics
        metrics_dict['total_loss'] = total_loss
        
        # Use only the current active stage for reported metrics
        # This ensures metrics reflect what's currently being trained
        try:
            current_stage_metrics = [m for m in stage_metrics_list if m['stage'] == current_active_stage]
            
            if current_stage_metrics:
                # Add the current stage metrics without stage prefix
                # These are the metrics that will be displayed in the progress bar
                metrics = current_stage_metrics[0]
                for key in ['base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard', 'reward']:
                    if key in metrics:
                        metrics_dict[key] = metrics[key]
                    elif key == 'loss_w_r' and 'total_loss' in metrics:
                        # Fallback: If loss_w_r not available, use total_loss
                        metrics_dict[key] = metrics['total_loss']
        except Exception as e:
            self.logger.error(f"Error aggregating stage metrics: {e}")
            # Add default metrics for display if aggregation fails
            device = target.device
            for key in ['base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']:
                metrics_dict[key] = torch.tensor(0.0, device=device)
            
        self.logger.debug(f"Total loss: {total_loss:.4f}")
        
        return metrics_dict