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
    
    def format_metrics_display(
        self,
        metrics: Dict[str, Any],
        ordered_metrics: Optional[List[str]] = None,
        show_stage_metrics: bool = True,
        active_stages: Optional[List[int]] = None
    ) -> Dict[str, str]:
        """
        Format metrics for display in progress bar with appropriate ordering.
        
        Args:
            metrics: Dictionary of metrics
            ordered_metrics: Optional list of metrics to prioritize in display
            show_stage_metrics: Whether to show per-stage metrics
            active_stages: Optional list of active stage indices (for curriculum learning)
            
        Returns:
            Dictionary of formatted metrics for display
        """
        display_dict = {}
        
        # Default ordered metrics if not provided
        if ordered_metrics is None:
            ordered_metrics = ['base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']
        
        # Add the direct metrics first - these are for the current active stage
        for metric_name in ordered_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                if torch.is_tensor(value):
                    value = value.item()
                
                # Validate value
                if np.isfinite(value) and abs(value) < 1e6:
                    # Use .4f precision for all metrics in the active stage
                    display_dict[metric_name] = f"{value:.4f}"
        
        # If active_stages is provided but empty, just return the basic metrics
        if active_stages is not None and len(active_stages) == 0:
            return display_dict
            
        # We don't need to show individual stage metrics in the progress bar
        # since we're already showing the active stage metrics above
        
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
    
    def get_epoch_averages(
        self,
        all_metrics: List[Dict[str, Any]],
        num_batches: int,
        prioritize_metrics: Optional[List[str]] = None,
        active_stages: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Calculate average metrics across all batches in an epoch.
        
        Args:
            all_metrics: List of metric dictionaries from all batches
            num_batches: Number of batches in the epoch
            prioritize_metrics: Optional list of metrics to prioritize in calculation
            active_stages: Optional list of active stage indices (for curriculum learning)
            
        Returns:
            Dictionary of averaged metrics
        """
        if not all_metrics:
            return {}
            
        avg_metrics = {}
        
        # Default prioritized metrics if not provided
        if prioritize_metrics is None:
            prioritize_metrics = ['total_loss', 'base_loss', 'loss_w_r', 'dice', 'precision', 'recall', 'jaccard']
        
        # Calculate averages for prioritized metrics first
        for metric_name in prioritize_metrics:
            values = []
            for m in all_metrics:
                if metric_name in m:
                    value = m[metric_name]
                    if torch.is_tensor(value):
                        value = value.item()
                    
                    if np.isfinite(value) and abs(value) < 1e9:
                        values.append(value)
            
            if values:
                avg_metrics[metric_name] = np.mean(values)
        
        # Handle stage-specific metrics
        # Identify all stage metrics in the data
        stage_metrics = {}
        
        for m in all_metrics:
            for key in m:
                if key.startswith('stage') and '_' in key:
                    parts = key.split('_')
                    stage_idx_str = parts[0].replace('stage', '')
                    
                    try:
                        stage_idx = int(stage_idx_str) - 1  # Convert to 0-based index
                        metric_name = '_'.join(parts[1:])
                        
                        # Skip stages not in active_stages if provided
                        if active_stages is not None and stage_idx not in active_stages:
                            continue
                            
                        if stage_idx not in stage_metrics:
                            stage_metrics[stage_idx] = {}
                            
                        if metric_name not in stage_metrics[stage_idx]:
                            stage_metrics[stage_idx][metric_name] = []
                            
                        value = m[key]
                        if torch.is_tensor(value):
                            value = value.item()
                            
                        if np.isfinite(value) and abs(value) < 1e9:
                            stage_metrics[stage_idx][metric_name].append(value)
                    except (ValueError, IndexError):
                        continue
        
        # Calculate averages for stage metrics
        for stage_idx, metrics in stage_metrics.items():
            for metric_name, values in metrics.items():
                if values:
                    key = f"stage{stage_idx+1}_{metric_name}"
                    avg_metrics[key] = np.mean(values)
        
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

class MetricAggregator:
    """Efficiently aggregates metrics during training with memory-optimized storage."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all aggregated values."""
        self.running_metrics = {}
        self.counts = {}
        
    def update(self, metrics: Dict[str, Any], count: int = 1):
        """Update running metrics."""
        for key, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
                
            # Convert to float if tensor
            if torch.is_tensor(value):
                value = value.item()
                
            # Skip invalid values
            if not np.isfinite(value):
                continue
                
            # Initialize if needed
            if key not in self.running_metrics:
                self.running_metrics[key] = 0.0
                self.counts[key] = 0
                
            # Update running sum and count
            self.running_metrics[key] += value * count
            self.counts[key] += count
            
    def get_averages(self) -> Dict[str, float]:
        """Get current average values."""
        averages = {}
        for key in self.running_metrics:
            if self.counts[key] > 0:
                avg = self.running_metrics[key] / self.counts[key]
                # Ensure the average is valid
                if np.isfinite(avg):
                    averages[key] = avg
                else:
                    averages[key] = 0.0
        return averages

    def get_stage_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Extract and organize stage-specific metrics into a structured dictionary.
        
        Returns:
            Dictionary with stage indices as keys and metric dictionaries as values
        """
        stage_metrics = {}
        
        # Extract metrics with stage prefixes
        for key, value in self.running_metrics.items():
            if key.startswith('stage') and '_' in key:
                parts = key.split('_', 1)
                stage_id = parts[0]
                metric_name = parts[1]
                
                # Get the stage number
                try:
                    stage_num = int(stage_id.replace('stage', ''))
                    
                    # Create stage entry if it doesn't exist
                    if stage_num not in stage_metrics:
                        stage_metrics[stage_num] = {}
                    
                    # Calculate average for this metric
                    if self.counts[key] > 0:
                        avg = value / self.counts[key]
                        if np.isfinite(avg):
                            stage_metrics[stage_num][metric_name] = avg
                        else:
                            stage_metrics[stage_num][metric_name] = 0.0
                except ValueError:
                    # Skip if we can't parse the stage number
                    continue
        
        return stage_metrics