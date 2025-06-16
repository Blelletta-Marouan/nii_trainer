"""
Reward-based loss functions for NII-Trainer.

This module implements advanced reward-based loss functions that incorporate
performance metrics as rewards to guide training towards better segmentation quality.
The reward system is particularly effective for cascaded segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from .segmentation import DiceLoss, FocalLoss


def hard_dice_coef(pred: torch.Tensor, target: torch.Tensor, 
                   threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes the hard Dice coefficient between predictions and targets.
    
    Args:
        pred: Predictions tensor of shape [B, 1, H, W] (probabilities)
        target: Target tensor of shape [B, 1, H, W] (binary)
        threshold: Threshold for binarizing predictions
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Dice coefficient as float
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    pred_sum = pred_bin.sum(dim=(1, 2, 3))
    target_sum = target_bin.sum(dim=(1, 2, 3))
    
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice.mean().item()


def hard_jaccard_index(pred: torch.Tensor, target: torch.Tensor,
                       threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes the hard Jaccard index (IoU) between predictions and targets.
    
    Args:
        pred: Predictions tensor of shape [B, 1, H, W] (probabilities)
        target: Target tensor of shape [B, 1, H, W] (binary)
        threshold: Threshold for binarizing predictions
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Jaccard index as float
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - intersection
    
    jaccard = (intersection + eps) / (union + eps)
    return jaccard.mean().item()


def hard_precision_metric(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes precision: TP / (TP + FP)
    
    Args:
        pred: Predictions tensor of shape [B, 1, H, W] (probabilities)
        target: Target tensor of shape [B, 1, H, W] (binary)
        threshold: Threshold for binarizing predictions
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Precision as float
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fp = pred_bin.sum(dim=(1, 2, 3)) - tp
    
    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()


def hard_recall_metric(pred: torch.Tensor, target: torch.Tensor,
                       threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes recall (sensitivity): TP / (TP + FN)
    
    Args:
        pred: Predictions tensor of shape [B, 1, H, W] (probabilities)
        target: Target tensor of shape [B, 1, H, W] (binary)
        threshold: Threshold for binarizing predictions
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Recall as float
    """
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fn = target_bin.sum(dim=(1, 2, 3)) - tp
    
    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean().item()


class RewardBasedLoss(nn.Module):
    """
    Base class for reward-based loss functions.
    
    This loss function combines traditional loss terms with reward terms
    computed from performance metrics to guide training towards better
    segmentation quality.
    
    Args:
        base_loss (nn.Module): Base loss function (e.g., DiceLoss, FocalLoss)
        reward_coefficient (float): Coefficient for reward term
        threshold (float): Threshold for computing hard metrics
    """
    
    def __init__(self, base_loss: nn.Module, reward_coefficient: float = 0.1,
                 threshold: float = 0.5):
        super(RewardBasedLoss, self).__init__()
        
        self.base_loss = base_loss
        self.reward_coefficient = reward_coefficient
        self.threshold = threshold
    
    def compute_reward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reward term based on performance metrics.
        To be implemented by subclasses.
        
        Args:
            pred: Predictions tensor (probabilities)
            target: Target tensor
            
        Returns:
            Reward tensor
        """
        raise NotImplementedError("Subclasses must implement compute_reward")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base loss and reward.
        
        Args:
            pred: Predictions tensor
            target: Target tensor
            
        Returns:
            Combined loss with reward
        """
        base_loss_value = self.base_loss(pred, target)
        reward_value = self.compute_reward(pred, target)
        
        return base_loss_value - self.reward_coefficient * reward_value


class DualOutputRewardLoss(nn.Module):
    """
    Advanced reward-based loss for dual-output cascaded models (e.g., liver-tumor segmentation).
    
    This loss function implements the sophisticated reward system from the original code,
    combining multiple metrics with different weights for different anatomical structures.
    
    The reward is computed from four metrics:
    - Dice coefficient  
    - Precision
    - Recall
    - Jaccard index (IoU)
    
    Each metric is computed separately for different outputs (e.g., liver and tumor) 
    and then combined with configurable weights.
    
    Args:
        weight_bce (float): Weight for BCE loss component
        weight_dice (float): Weight for Dice loss component  
        liver_weight (float): Positive class weight for liver segmentation
        tumor_weight (float): Positive class weight for tumor segmentation
        gamma (float): Focusing parameter for Focal loss
        reward_coefficient (float): Coefficient for reward term
        threshold_liver (float): Threshold for liver predictions
        threshold_tumor (float): Threshold for tumor predictions
    """
    
    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5,
                 liver_weight: float = 5.0, tumor_weight: float = 5.0,
                 gamma: float = 2.0, reward_coefficient: float = 0.1,
                 threshold_liver: float = 0.5, threshold_tumor: float = 0.5):
        super(DualOutputRewardLoss, self).__init__()
        
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.liver_weight = liver_weight
        self.tumor_weight = tumor_weight
        self.reward_coefficient = reward_coefficient
        self.threshold_liver = threshold_liver
        self.threshold_tumor = threshold_tumor
        
        # BCE loss for liver segmentation with class weighting
        self.focal_liver = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.liver_weight])
        )
        
        # Focal loss for tumor segmentation
        self.focal_tumor = FocalLoss(alpha=self.tumor_weight, gamma=gamma, logits=True)
        
        # Dice loss
        self.dice_loss = DiceLoss()
    
    def forward(self, liver_pred: torch.Tensor, tumor_pred: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dual-output model with reward.
        
        Args:
            liver_pred: Liver predictions tensor [B, 1, H, W] (logits)
            tumor_pred: Tumor predictions tensor [B, 1, H, W] (logits)
            y: Ground truth tensor [B, H, W] with values 0=background, 1=liver, 2=tumor
            
        Returns:
            Combined loss with reward term
        """
        device = liver_pred.device
        
        # Move pos_weight to correct device
        if self.focal_liver.pos_weight.device != device:
            self.focal_liver.pos_weight = self.focal_liver.pos_weight.to(device)
        
        # Create binary ground truths with channel dimension [B, 1, H, W]
        liver_gt = (y == 1).float().unsqueeze(1)
        tumor_gt = (y == 2).float().unsqueeze(1)
        
        # Compute base losses
        loss_liver = self.focal_liver(liver_pred, liver_gt)
        loss_tumor = self.focal_tumor(tumor_pred, tumor_gt)
        
        # Dice losses
        dice_loss_liver = self.dice_loss(torch.sigmoid(liver_pred), liver_gt)
        dice_loss_tumor = self.dice_loss(torch.sigmoid(tumor_pred), tumor_gt)
        
        # Convert Dice loss to Dice coefficient (metric for reward)
        dice_liver_metric = 1 - dice_loss_liver
        dice_tumor_metric = 1 - dice_loss_tumor
        
        # Compute hard metrics using sigmoid probabilities
        liver_prob = torch.sigmoid(liver_pred)
        tumor_prob = torch.sigmoid(tumor_pred)
        
        recall_liver = hard_recall_metric(liver_prob, liver_gt, threshold=self.threshold_liver)
        recall_tumor = hard_recall_metric(tumor_prob, tumor_gt, threshold=self.threshold_tumor)
        
        precision_liver = hard_precision_metric(liver_prob, liver_gt, threshold=self.threshold_liver)  
        precision_tumor = hard_precision_metric(tumor_prob, tumor_gt, threshold=self.threshold_tumor)
        
        jaccard_liver = hard_jaccard_index(liver_prob, liver_gt, threshold=self.threshold_liver)
        jaccard_tumor = hard_jaccard_index(tumor_prob, tumor_gt, threshold=self.threshold_tumor)
        
        # Combine per-branch metrics with 40% weight for liver and 60% for tumor
        dice_reward = 0.4 * dice_liver_metric + 0.6 * dice_tumor_metric
        
        precision_reward = 0.4 * torch.tensor(precision_liver, device=device) + \
                          0.6 * torch.tensor(precision_tumor, device=device)
        
        recall_reward = 0.4 * torch.tensor(recall_liver, device=device) + \
                       0.6 * torch.tensor(recall_tumor, device=device)
        
        jaccard_reward = 0.4 * torch.tensor(jaccard_liver, device=device) + \
                        0.6 * torch.tensor(jaccard_tumor, device=device)
        
        # Compute overall reward with weighted composition:
        # 20% dice, 20% precision, 20% recall, and 40% jaccard
        reward = self.reward_coefficient * (
            0.2 * dice_reward + 
            0.2 * precision_reward +
            0.2 * recall_reward + 
            0.4 * jaccard_reward
        )
        
        # Combine base losses
        base_loss = (self.weight_bce * (loss_liver + loss_tumor) +
                    self.weight_dice * (dice_loss_liver + dice_loss_tumor))
        
        # Final loss with reward (subtract reward to minimize loss)
        total_loss = base_loss - reward
        
        return total_loss


class MultiMetricRewardLoss(nn.Module):
    """
    Multi-metric reward loss that can combine arbitrary metrics as rewards.
    
    Args:
        base_loss (nn.Module): Base loss function
        metric_weights (Dict[str, float]): Dictionary of metric name -> weight
        reward_coefficient (float): Overall reward coefficient
        threshold (float): Threshold for computing hard metrics
    """
    
    def __init__(self, base_loss: nn.Module, metric_weights: Dict[str, float],
                 reward_coefficient: float = 0.1, threshold: float = 0.5):
        super(MultiMetricRewardLoss, self).__init__()
        
        self.base_loss = base_loss
        self.metric_weights = metric_weights
        self.reward_coefficient = reward_coefficient
        self.threshold = threshold
        
        # Normalize weights
        total_weight = sum(metric_weights.values())
        self.metric_weights = {k: v / total_weight for k, v in metric_weights.items()}
    
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            pred: Predictions tensor (probabilities)
            target: Target tensor
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        if 'dice' in self.metric_weights:
            metrics['dice'] = hard_dice_coef(pred, target, self.threshold)
        
        if 'jaccard' in self.metric_weights:
            metrics['jaccard'] = hard_jaccard_index(pred, target, self.threshold)
        
        if 'precision' in self.metric_weights:
            metrics['precision'] = hard_precision_metric(pred, target, self.threshold)
        
        if 'recall' in self.metric_weights:
            metrics['recall'] = hard_recall_metric(pred, target, self.threshold)
        
        return metrics
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-metric reward.
        
        Args:
            pred: Predictions tensor (logits)
            target: Target tensor
            
        Returns:
            Loss with multi-metric reward
        """
        # Compute base loss
        base_loss_value = self.base_loss(pred, target)
        
        # Compute probabilities for metric calculation
        if pred.dim() == target.dim() + 1:  # pred has channel dim, target doesn't
            pred_prob = torch.sigmoid(pred)
            target_expanded = target.unsqueeze(1).float()
        else:
            pred_prob = torch.sigmoid(pred)
            target_expanded = target.float()
        
        # Compute metrics
        metrics = self.compute_metrics(pred_prob, target_expanded)
        
        # Compute weighted reward
        reward = 0.0
        for metric_name, metric_value in metrics.items():
            if metric_name in self.metric_weights:
                weight = self.metric_weights[metric_name]
                reward += weight * torch.tensor(metric_value, device=pred.device)
        
        # Apply reward coefficient and subtract from loss
        total_loss = base_loss_value - self.reward_coefficient * reward
        
        return total_loss


class ProgressiveRewardLoss(nn.Module):
    """
    Progressive reward loss that increases reward influence during training.
    
    Args:
        base_loss (nn.Module): Base loss function
        reward_loss (nn.Module): Reward-based loss function
        max_reward_coef (float): Maximum reward coefficient
        warmup_steps (int): Number of steps for reward warmup
    """
    
    def __init__(self, base_loss: nn.Module, reward_loss: nn.Module,
                 max_reward_coef: float = 0.2, warmup_steps: int = 1000):
        super(ProgressiveRewardLoss, self).__init__()
        
        self.base_loss = base_loss
        self.reward_loss = reward_loss
        self.max_reward_coef = max_reward_coef
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with progressive reward scaling.
        
        Args:
            pred: Predictions tensor
            target: Target tensor
            
        Returns:
            Loss with progressive reward scaling
        """
        # Compute current reward coefficient
        progress = min(1.0, self.current_step / self.warmup_steps)
        current_reward_coef = self.max_reward_coef * progress
        
        # Update reward coefficient in reward loss if it has one
        if hasattr(self.reward_loss, 'reward_coefficient'):
            original_coef = self.reward_loss.reward_coefficient
            self.reward_loss.reward_coefficient = current_reward_coef
        
        # Compute losses
        base_loss_value = self.base_loss(pred, target)
        reward_loss_value = self.reward_loss(pred, target)
        
        # Restore original coefficient
        if hasattr(self.reward_loss, 'reward_coefficient'):
            self.reward_loss.reward_coefficient = original_coef
        
        # Combine with progressive weighting
        total_loss = (1 - progress) * base_loss_value + progress * reward_loss_value
        
        self.current_step += 1
        return total_loss