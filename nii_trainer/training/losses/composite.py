"""
Composite loss functions for NII-Trainer.

This module implements composite loss functions that combine multiple loss types
for more effective training, especially useful for cascaded segmentation models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable
from .segmentation import DiceLoss, FocalLoss, TverskyLoss, IoULoss


class WeightedCombinedLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.
    
    Args:
        losses (Dict[str, nn.Module]): Dictionary of loss name -> loss function
        weights (Dict[str, float]): Dictionary of loss name -> weight
        normalize_weights (bool): Whether to normalize weights to sum to 1
    """
    
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float],
                 normalize_weights: bool = True):
        super(WeightedCombinedLoss, self).__init__()
        
        self.losses = nn.ModuleDict(losses)
        self.weights = weights.copy()
        
        if normalize_weights:
            total_weight = sum(weights.values())
            self.weights = {k: v / total_weight for k, v in weights.items()}
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining all losses.
        
        Args:
            pred: Predictions tensor
            target: Target tensor
            
        Returns:
            Combined weighted loss
        """
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            if name in self.weights:
                loss_value = loss_fn(pred, target)
                total_loss += self.weights[name] * loss_value
        
        return total_loss


class DiceBCELoss(nn.Module):
    """
    Combination of Dice Loss and Binary Cross Entropy Loss.
    
    Args:
        dice_weight (float): Weight for Dice loss component
        bce_weight (float): Weight for BCE loss component  
        smooth (float): Smoothing factor for Dice loss
        pos_weight (Optional[torch.Tensor]): Positive class weight for BCE
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 smooth: float = 1.0, pos_weight: Optional[torch.Tensor] = None):
        super(DiceBCELoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(torch.sigmoid(pred), target)
        bce_loss = self.bce_loss(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class DiceFocalLoss(nn.Module):
    """
    Combination of Dice Loss and Focal Loss.
    
    Args:
        dice_weight (float): Weight for Dice loss component
        focal_weight (float): Weight for Focal loss component
        alpha (float): Alpha parameter for Focal loss
        gamma (float): Gamma parameter for Focal loss
        smooth (float): Smoothing factor for Dice loss
    """
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 alpha: float = 1.0, gamma: float = 2.0, smooth: float = 1.0):
        super(DiceFocalLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, logits=True)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(torch.sigmoid(pred), target)
        focal_loss = self.focal_loss(pred, target)
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class MultiStageLoss(nn.Module):
    """
    Loss function for multi-stage/cascaded models.
    
    Applies different loss functions and weights to different stages,
    with optional deep supervision.
    
    Args:
        stage_losses (Dict[int, nn.Module]): Stage index -> loss function
        stage_weights (Dict[int, float]): Stage index -> weight
        deep_supervision (bool): Whether to apply deep supervision
    """
    
    def __init__(self, stage_losses: Dict[int, nn.Module], 
                 stage_weights: Dict[int, float],
                 deep_supervision: bool = True):
        super(MultiStageLoss, self).__init__()
        
        self.stage_losses = nn.ModuleDict({str(k): v for k, v in stage_losses.items()})
        self.stage_weights = stage_weights
        self.deep_supervision = deep_supervision
    
    def forward(self, stage_preds: List[torch.Tensor], 
                targets: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass for multi-stage predictions.
        
        Args:
            stage_preds: List of predictions from each stage
            targets: Target tensor(s) - single tensor or list per stage
            
        Returns:
            Combined multi-stage loss
        """
        total_loss = 0.0
        
        # Handle single target for all stages
        if isinstance(targets, torch.Tensor):
            targets = [targets] * len(stage_preds)
        
        for stage_idx, pred in enumerate(stage_preds):
            stage_key = str(stage_idx)
            
            if stage_key in self.stage_losses and stage_idx < len(targets):
                loss_fn = self.stage_losses[stage_key]
                target = targets[stage_idx]
                
                stage_loss = loss_fn(pred, target)
                weight = self.stage_weights.get(stage_idx, 1.0)
                
                # Apply deep supervision weighting (higher weight for later stages)
                if self.deep_supervision:
                    supervision_weight = (stage_idx + 1) / len(stage_preds)
                    weight *= supervision_weight
                
                total_loss += weight * stage_loss
        
        return total_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.
    
    Args:
        base_loss (nn.Module): Base loss function
        adaptation_strategy (str): Strategy for adaptation ('linear', 'exponential', 'cosine')
        adaptation_factor (float): Factor controlling adaptation rate
    """
    
    def __init__(self, base_loss: nn.Module, adaptation_strategy: str = 'linear',
                 adaptation_factor: float = 1.0):
        super(AdaptiveLoss, self).__init__()
        
        self.base_loss = base_loss
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_factor = adaptation_factor
        self.training_step = 0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss_value = self.base_loss(pred, target)
        
        # Calculate adaptation weight
        if self.adaptation_strategy == 'linear':
            weight = 1.0 + self.adaptation_factor * (self.training_step / 1000.0)
        elif self.adaptation_strategy == 'exponential':
            weight = torch.exp(torch.tensor(self.adaptation_factor * self.training_step / 1000.0))
        elif self.adaptation_strategy == 'cosine':
            weight = 1.0 + self.adaptation_factor * torch.cos(torch.tensor(self.training_step / 1000.0))
        else:
            weight = 1.0
        
        self.training_step += 1
        return weight * base_loss_value


class BalancedLoss(nn.Module):
    """
    Automatically balanced loss that adjusts weights based on class frequencies.
    
    Args:
        base_loss (nn.Module): Base loss function
        balance_strategy (str): Balancing strategy ('inverse_freq', 'focal_weight')
        smooth_factor (float): Smoothing factor for frequency calculation
    """
    
    def __init__(self, base_loss: nn.Module, balance_strategy: str = 'inverse_freq',
                 smooth_factor: float = 1.0):
        super(BalancedLoss, self).__init__()
        
        self.base_loss = base_loss
        self.balance_strategy = balance_strategy
        self.smooth_factor = smooth_factor
        self.class_frequencies = None
    
    def _update_frequencies(self, target: torch.Tensor):
        """Update class frequency estimates."""
        batch_freq = torch.bincount(target.flatten().long(), minlength=2).float()
        
        if self.class_frequencies is None:
            self.class_frequencies = batch_freq
        else:
            # Exponential moving average
            alpha = 0.1
            self.class_frequencies = (1 - alpha) * self.class_frequencies + alpha * batch_freq
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._update_frequencies(target)
        
        base_loss_value = self.base_loss(pred, target)
        
        if self.class_frequencies is not None and self.balance_strategy == 'inverse_freq':
            # Calculate inverse frequency weights
            total_samples = self.class_frequencies.sum()
            weights = total_samples / (self.class_frequencies + self.smooth_factor)
            
            # Apply class-specific weighting (simplified for binary case)
            pos_weight = weights[1] / weights[0] if len(weights) > 1 else 1.0
            base_loss_value = base_loss_value * pos_weight
        
        return base_loss_value