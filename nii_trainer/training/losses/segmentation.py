"""
Segmentation loss functions for NII-Trainer.

This module implements various loss functions commonly used in medical image
segmentation tasks, including Dice loss, Focal loss, and other variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation.
    
    The Dice loss is computed as 1 - Dice coefficient, where the Dice coefficient
    measures the overlap between predicted and target masks.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0
        reduction (str): Type of reduction to apply. Options: 'mean', 'sum', 'none'
        square_denominator (bool): Whether to square the denominator terms
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean', 
                 square_denominator: bool = False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.square_denominator = square_denominator

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Dice loss.
        
        Args:
            pred: Predictions tensor of shape [B, C, H, W] or [B, H, W]
            target: Target tensor of shape [B, C, H, W] or [B, H, W]
            
        Returns:
            Dice loss value
        """
        # Ensure tensors have the same shape
        pred = pred.contiguous()
        target = target.contiguous()
        
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        if self.square_denominator:
            denominator = pred_sum.pow(2) + target_sum.pow(2)
        else:
            denominator = pred_sum + target_sum
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        # Convert to loss (1 - dice)
        loss = 1 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation tasks.
    
    The Focal Loss down-weights easy examples and focuses learning on hard examples.
    
    Args:
        alpha (float): Weighting factor for rare class. Default: 1.0
        gamma (float): Focusing parameter. Default: 2.0
        logits (bool): Whether inputs are logits or probabilities. Default: True
        reduction (str): Type of reduction to apply. Options: 'mean', 'sum', 'none'
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 logits: bool = True, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal loss.
        
        Args:
            inputs: Predictions tensor
            targets: Target tensor
            
        Returns:
            Focal loss value
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with different penalties for FP and FN.
    
    Args:
        alpha (float): Weight for false positives. Default: 0.3
        beta (float): Weight for false negatives. Default: 0.7
        smooth (float): Smoothing factor. Default: 1.0
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = ((1 - target_flat) * pred_flat).sum(dim=1)
        fn = (target_flat * (1 - pred_flat)).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return (1 - tversky).mean()


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.
    
    Args:
        smooth (float): Smoothing factor. Default: 1.0
    """
    
    def __init__(self, smooth: float = 1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return (1 - iou).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better boundary prediction in segmentation.
    
    Args:
        theta0 (float): Boundary width parameter. Default: 3.0
        theta (float): Boundary weight parameter. Default: 5.0
    """
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simplified boundary loss implementation.
        For full implementation, distance transforms would be needed.
        """
        # This is a simplified version - full implementation would require
        # distance transform computations
        pred_boundary = self._get_boundary(pred)
        target_boundary = self._get_boundary(target)
        
        return F.mse_loss(pred_boundary, target_boundary)
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary using morphological operations."""
        # Simplified boundary extraction using gradient
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                      dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        if len(mask.shape) == 4:  # [B, C, H, W]
            boundary = F.conv2d(mask, laplacian_kernel, padding=1)
        else:  # [B, H, W]
            boundary = F.conv2d(mask.unsqueeze(1), laplacian_kernel, padding=1).squeeze(1)
        
        return torch.abs(boundary)