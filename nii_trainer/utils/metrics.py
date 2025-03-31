from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd

def calculate_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Calculate per-class metrics including:
    - Precision
    - Recall
    - F1 Score
    - IoU (Jaccard)
    Returns a pandas DataFrame with metrics per class.
    """
    metrics_dict = {
        'Class': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'IoU': [],
        'Support': []
    }
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
        
    # Flatten arrays
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Calculate metrics for each class
    for i, class_name in enumerate(class_names):
        true_positive = np.sum((predictions == i) & (targets == i))
        false_positive = np.sum((predictions == i) & (targets != i))
        false_negative = np.sum((predictions != i) & (targets == i))
        
        precision = true_positive / (true_positive + false_positive + 1e-7)
        recall = true_positive / (true_positive + false_negative + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-7)
        support = np.sum(targets == i)
        
        metrics_dict['Class'].append(class_name)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1'].append(f1)
        metrics_dict['IoU'].append(iou)
        metrics_dict['Support'].append(support)
        
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    
    # Add mean metrics (excluding background class)
    means = metrics_df.iloc[1:].mean(numeric_only=True)
    means_dict = {'Class': 'Mean (No BG)'}
    means_dict.update({col: means[col] for col in means.index})
    metrics_df = pd.concat([metrics_df, pd.DataFrame([means_dict])], ignore_index=True)
    
    return metrics_df

def calculate_volumetric_metrics(
    pred_slices: List[np.ndarray],
    target_slices: List[np.ndarray],
    spacing: Tuple[float, float, float]
) -> Dict[str, float]:
    """
    Calculate 3D volumetric metrics when dealing with slices from a volume.
    Args:
        pred_slices: List of predicted segmentation slices
        target_slices: List of ground truth segmentation slices
        spacing: Physical spacing (x, y, z) in mm
    Returns:
        Dictionary with volumetric metrics
    """
    # Stack slices into volumes
    pred_volume = np.stack(pred_slices, axis=0)
    target_volume = np.stack(target_slices, axis=0)
    
    # Calculate volume in mm³ (multiply by physical spacing)
    voxel_volume = np.prod(spacing)
    pred_volume_mm3 = np.sum(pred_volume > 0) * voxel_volume
    target_volume_mm3 = np.sum(target_volume > 0) * voxel_volume
    
    # Calculate volumetric overlap
    intersection = np.sum((pred_volume > 0) & (target_volume > 0)) * voxel_volume
    union = np.sum((pred_volume > 0) | (target_volume > 0)) * voxel_volume
    
    # Calculate metrics
    volumetric_dice = (2 * intersection) / (pred_volume_mm3 + target_volume_mm3 + 1e-7)
    volumetric_iou = intersection / (union + 1e-7)
    volume_difference = abs(pred_volume_mm3 - target_volume_mm3)
    volume_error = volume_difference / target_volume_mm3
    
    return {
        'volumetric_dice': volumetric_dice,
        'volumetric_iou': volumetric_iou,
        'volume_difference_mm3': volume_difference,
        'volume_error': volume_error,
        'predicted_volume_mm3': pred_volume_mm3,
        'target_volume_mm3': target_volume_mm3
    }