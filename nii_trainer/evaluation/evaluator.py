"""
Evaluation framework for NII-Trainer.

This module provides comprehensive evaluation capabilities including
metrics computation, statistical analysis, and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import pandas as pd

from ..core.exceptions import EvaluationError
from ..core.registry import register_evaluator
from ..core.config import EvaluationConfig, GlobalConfig


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, config: EvaluationConfig, **kwargs):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics = {}
        self.predictions = []
        self.ground_truths = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation."""
        logger = logging.getLogger(f"nii_trainer.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def evaluate(self, model: nn.Module, data_loader, device: torch.device) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        pass
    
    def reset(self) -> None:
        """Reset evaluator state."""
        self.metrics = {}
        self.predictions = []
        self.ground_truths = []


@register_evaluator('segmentation')
class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation tasks."""
    
    def __init__(self, config: EvaluationConfig, num_classes: int = 2, **kwargs):
        super().__init__(config, **kwargs)
        self.num_classes = num_classes
        self.threshold = 0.5
        
    def evaluate(self, model: nn.Module, data_loader, device: torch.device) -> Dict[str, Any]:
        """Evaluate segmentation model."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images)
                if isinstance(outputs, list):
                    outputs = outputs[-1]  # Use final stage output
                
                # Apply sigmoid/softmax
                if self.num_classes == 2:
                    predictions = torch.sigmoid(outputs)
                else:
                    predictions = torch.softmax(outputs, dim=1)
                
                # Store for metrics computation
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        
        # Threshold optimization if enabled
        if self.config.threshold_optimization:
            best_threshold, threshold_metrics = self._optimize_threshold(predictions, targets)
            metrics['best_threshold'] = best_threshold
            metrics.update(threshold_metrics)
        
        # Statistical analysis if enabled
        if self.config.compute_confidence_intervals:
            ci_metrics = self._compute_confidence_intervals(predictions, targets)
            metrics.update(ci_metrics)
        
        self.logger.info("Evaluation completed successfully!")
        return metrics
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation metrics."""
        metrics = {}
        
        # Binarize predictions for binary segmentation
        if self.num_classes == 2:
            pred_binary = (predictions > self.threshold).float()
            targets_binary = targets.float()
        else:
            pred_binary = torch.argmax(predictions, dim=1).float()
            targets_binary = targets.float()
        
        # Compute basic metrics for each class
        for class_idx in range(self.num_classes if self.num_classes > 2 else 1):
            if self.num_classes == 2:
                pred_class = pred_binary
                target_class = targets_binary
                class_name = "foreground"
            else:
                pred_class = (pred_binary == class_idx).float()
                target_class = (targets_binary == class_idx).float()
                class_name = f"class_{class_idx}"
            
            # Compute per-class metrics
            dice = self._compute_dice(pred_class, target_class)
            iou = self._compute_iou(pred_class, target_class)
            precision = self._compute_precision(pred_class, target_class)
            recall = self._compute_recall(pred_class, target_class)
            specificity = self._compute_specificity(pred_class, target_class)
            
            metrics[f'dice_{class_name}'] = dice
            metrics[f'iou_{class_name}'] = iou
            metrics[f'precision_{class_name}'] = precision
            metrics[f'recall_{class_name}'] = recall
            metrics[f'specificity_{class_name}'] = specificity
        
        # Compute average metrics
        if self.num_classes > 2:
            for metric_name in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                class_metrics = [v for k, v in metrics.items() if k.startswith(f'{metric_name}_class_')]
                metrics[f'{metric_name}_mean'] = np.mean(class_metrics)
        
        # Compute additional metrics if requested
        if 'accuracy' in self.config.metrics:
            metrics['accuracy'] = self._compute_accuracy(pred_binary, targets_binary)
        
        if self.config.compute_hausdorff:
            hausdorff_dist = self._compute_hausdorff_distance(pred_binary, targets_binary)
            metrics['hausdorff_distance'] = hausdorff_dist
        
        if self.config.compute_surface_distance:
            surface_dist = self._compute_surface_distance(pred_binary, targets_binary)
            metrics['surface_distance'] = surface_dist
        
        return metrics
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute Dice coefficient."""
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute Intersection over Union."""
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def _compute_precision(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute precision."""
        true_positive = torch.sum(pred * target)
        predicted_positive = torch.sum(pred)
        precision = (true_positive + smooth) / (predicted_positive + smooth)
        return precision.item()
    
    def _compute_recall(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute recall (sensitivity)."""
        true_positive = torch.sum(pred * target)
        actual_positive = torch.sum(target)
        recall = (true_positive + smooth) / (actual_positive + smooth)
        return recall.item()
    
    def _compute_specificity(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute specificity."""
        true_negative = torch.sum((1 - pred) * (1 - target))
        actual_negative = torch.sum(1 - target)
        specificity = (true_negative + smooth) / (actual_negative + smooth)
        return specificity.item()
    
    def _compute_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute pixel-wise accuracy."""
        correct = torch.sum(pred == target)
        total = torch.numel(pred)
        accuracy = correct.float() / total
        return accuracy.item()
    
    def _compute_hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Hausdorff distance (simplified implementation)."""
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # Convert to numpy and get boundary points
            pred_np = pred.numpy().astype(bool)
            target_np = target.numpy().astype(bool)
            
            # Find boundary points (simplified)
            pred_points = np.argwhere(pred_np)
            target_points = np.argwhere(target_np)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Compute directed Hausdorff distances
            dist1 = directed_hausdorff(pred_points, target_points)[0]
            dist2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(dist1, dist2)
        except ImportError:
            self.logger.warning("scipy not available for Hausdorff distance computation")
            return 0.0
    
    def _compute_surface_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute average surface distance (simplified implementation)."""
        # This is a simplified implementation
        # For production use, consider using specialized libraries like SimpleITK
        try:
            from scipy import ndimage
            
            pred_np = pred.numpy().astype(bool)
            target_np = target.numpy().astype(bool)
            
            # Compute distance transforms
            pred_dist = ndimage.distance_transform_edt(~pred_np)
            target_dist = ndimage.distance_transform_edt(~target_np)
            
            # Compute surface distances
            pred_surface = ndimage.binary_erosion(pred_np) ^ pred_np
            target_surface = ndimage.binary_erosion(target_np) ^ target_np
            
            if not np.any(pred_surface) or not np.any(target_surface):
                return 0.0
            
            distances = np.concatenate([
                pred_dist[target_surface],
                target_dist[pred_surface]
            ])
            
            return np.mean(distances)
        except ImportError:
            self.logger.warning("scipy not available for surface distance computation")
            return 0.0
    
    def _optimize_threshold(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Optimize threshold for best performance."""
        thresholds = np.linspace(
            self.config.threshold_range[0], 
            self.config.threshold_range[1], 
            self.config.threshold_steps
        )
        
        best_threshold = 0.5
        best_metric = 0.0
        threshold_results = {}
        
        metric_name = self.config.threshold_metric
        
        for threshold in thresholds:
            # Apply threshold
            pred_binary = (predictions > threshold).float()
            
            # Compute metric
            if metric_name == 'dice':
                metric_value = self._compute_dice(pred_binary, targets.float())
            elif metric_name == 'iou':
                metric_value = self._compute_iou(pred_binary, targets.float())
            elif metric_name == 'precision':
                metric_value = self._compute_precision(pred_binary, targets.float())
            elif metric_name == 'recall':
                metric_value = self._compute_recall(pred_binary, targets.float())
            else:
                metric_value = self._compute_dice(pred_binary, targets.float())
            
            threshold_results[f'threshold_{threshold:.3f}_{metric_name}'] = metric_value
            
            if metric_value > best_metric:
                best_metric = metric_value
                best_threshold = threshold
        
        return best_threshold, threshold_results
    
    def _compute_confidence_intervals(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Compute confidence intervals using bootstrap sampling."""
        n_samples = len(predictions)
        n_bootstrap = self.config.bootstrap_samples
        confidence_level = self.config.confidence_level
        
        bootstrap_metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': []
        }
        
        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_sample = predictions[indices]
            target_sample = targets[indices]
            
            # Compute metrics for sample
            pred_binary = (pred_sample > self.threshold).float()
            target_binary = target_sample.float()
            
            bootstrap_metrics['dice'].append(self._compute_dice(pred_binary, target_binary))
            bootstrap_metrics['iou'].append(self._compute_iou(pred_binary, target_binary))
            bootstrap_metrics['precision'].append(self._compute_precision(pred_binary, target_binary))
            bootstrap_metrics['recall'].append(self._compute_recall(pred_binary, target_binary))
        
        # Compute confidence intervals
        ci_results = {}
        alpha = 1 - confidence_level
        
        for metric_name, values in bootstrap_metrics.items():
            values = np.array(values)
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            
            ci_results[f'{metric_name}_ci_lower'] = lower
            ci_results[f'{metric_name}_ci_upper'] = upper
            ci_results[f'{metric_name}_ci_mean'] = np.mean(values)
            ci_results[f'{metric_name}_ci_std'] = np.std(values)
        
        return ci_results


class MetricsTracker:
    """Track and aggregate metrics across multiple evaluations."""
    
    def __init__(self):
        self.metrics_history = []
        self.aggregated_metrics = {}
    
    def add_metrics(self, metrics: Dict[str, float], metadata: Dict[str, Any] = None) -> None:
        """Add metrics from one evaluation."""
        entry = {
            'metrics': metrics.copy(),
            'metadata': metadata or {}
        }
        self.metrics_history.append(entry)
    
    def compute_aggregated_metrics(self) -> Dict[str, Any]:
        """Compute aggregated statistics across all evaluations."""
        if not self.metrics_history:
            return {}
        
        # Get all metric names
        all_metric_names = set()
        for entry in self.metrics_history:
            all_metric_names.update(entry['metrics'].keys())
        
        aggregated = {}
        
        for metric_name in all_metric_names:
            values = []
            for entry in self.metrics_history:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
            
            if values:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)
                aggregated[f'{metric_name}_median'] = np.median(values)
        
        return aggregated
    
    def save_results(self, output_path: str) -> None:
        """Save evaluation results to file."""
        results = {
            'individual_metrics': self.metrics_history,
            'aggregated_metrics': self.compute_aggregated_metrics()
        }
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Save as CSV
            df = pd.DataFrame([entry['metrics'] for entry in self.metrics_history])
            df.to_csv(output_path, index=False)


def create_evaluator(evaluator_name: str, **kwargs):
    """Factory function to create evaluators."""
    from ..core.registry import EVALUATORS
    
    if not EVALUATORS.has(evaluator_name):
        raise EvaluationError(f"Unknown evaluator: {evaluator_name}")
    
    evaluator_class = EVALUATORS.get(evaluator_name)
    return evaluator_class(**kwargs)