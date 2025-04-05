from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from tqdm import tqdm

class MetricsManager:
    """
    Manages tracking, aggregation, and reporting of training metrics.
    Abstracts metric calculation, storage, and visualization.
    """
    
    def __init__(
        self,
        classes: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics manager.
        
        Args:
            classes: List of class names
            logger: Optional logger for logging metrics
        """
        self.classes = classes
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_base_loss': [], 'val_base_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_precision': [], 'val_precision': [],
            'train_jaccard': [], 'val_jaccard': [],
            'train_recall': [], 'val_recall': [],
            'train_metrics': [], 'val_metrics': []
        }
        
    def update_history(self, phase: str, metrics: Dict[str, float]):
        """
        Update metrics history for a given phase.
        
        Args:
            phase: 'train' or 'val'
            metrics: Dictionary of metrics
        """
        self.metrics_history[f'{phase}_loss'].append(metrics.get('loss', 0))
        
        if 'base_loss' in metrics:
            self.metrics_history[f'{phase}_base_loss'].append(metrics.get('base_loss', 0))
            
        # Store main metrics separately for easier plotting
        self.metrics_history[f'{phase}_dice'].append(metrics.get('dice', 0))
        self.metrics_history[f'{phase}_precision'].append(metrics.get('precision', 0))
        self.metrics_history[f'{phase}_jaccard'].append(metrics.get('iou', 0))  # Jaccard = IoU
        self.metrics_history[f'{phase}_recall'].append(metrics.get('recall', 0))
        
        # Store all metrics for reference
        self.metrics_history[f'{phase}_metrics'].append(metrics)
    
    def format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Format metrics for display in progress bar.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of formatted metrics for display
        """
        display_dict = {}
        
        # Order and display names for the metrics
        metric_mappings = {
            'base_loss': 'base_loss',
            'loss_w_r': 'loss_w_r',  # Use the direct loss_w_r value
            'dice': 'dice',
            'precision': 'precision',
            'recall': 'recall', 
            'iou': 'jaccard'
        }
        
        for key, display_key in metric_mappings.items():
            if key in metrics:
                # Get value and ensure it's a float
                value = metrics[key]
                if torch.is_tensor(value):
                    value = value.item()
                
                # Validate the value
                if not np.isfinite(value) or abs(value) > 1e6:
                    value = 0.0
                
                # Format with 4 decimal places
                display_dict[display_key] = f"{value:.4f}"
                
        return display_dict
    
    def average_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Average a list of metric dictionaries.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary of averaged metrics
        """
        if not metrics_list:
            return {}
            
        avg_metrics = {}
        
        # First, handle direct metrics without stage prefixes
        for key in metrics_list[0].keys():
            # Skip non-numeric metrics and ones with stage prefixes
            if key.startswith('stage') or not self._is_numeric_metric(metrics_list[0][key]):
                continue
                
            avg_metrics[key] = self._safe_mean([m[key] for m in metrics_list if key in m])
                
        # Process stage-specific metrics if needed
        current_stage_metrics = self._extract_current_stage_metrics(metrics_list)
        avg_metrics.update(current_stage_metrics)
                
        return avg_metrics
    
    def _is_numeric_metric(self, value: Any) -> bool:
        """Check if a value is a numeric metric that can be averaged."""
        if not isinstance(value, (int, float, np.number)) and not torch.is_tensor(value):
            return False
            
        if torch.is_tensor(value):
            try:
                value = float(value.item())
            except (ValueError, RuntimeError):
                return False
                
        return np.isfinite(value) and abs(value) < 1e9
    
    def _safe_mean(self, values: List[Any]) -> float:
        """Safely calculate the mean of a list of values."""
        try:
            # Convert tensor values to float
            float_values = []
            for v in values:
                if torch.is_tensor(v):
                    float_values.append(float(v.item()))
                else:
                    float_values.append(float(v))
            
            # Filter out invalid values
            valid_values = [v for v in float_values if np.isfinite(v) and abs(v) < 1e9]
            
            if not valid_values:
                return 0.0
                
            return float(np.mean(valid_values))
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_current_stage_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract metrics for the current active stage."""
        # This simplified version just uses the metrics that don't have stage prefixes
        return {}