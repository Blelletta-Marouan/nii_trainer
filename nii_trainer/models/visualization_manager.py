from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from ..visualization.visualizer import SegmentationVisualizer

class VisualizationManager:
    """
    Manages visualization of metrics, predictions, and model performance.
    Handles saving visualizations to disk and generating summary figures.
    """
    
    def __init__(
        self,
        classes: List[str],
        save_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the visualization manager.
        
        Args:
            classes: List of class names
            save_dir: Directory to save visualizations
            experiment_name: Optional experiment name
            logger: Optional logger for logging information
        """
        self.classes = classes
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name or "experiment"
        self.logger = logger or logging.getLogger(__name__)
        
        # Create required directories
        self.visualizations_dir = self.save_dir / "visualizations"
        self.metrics_dir = self.save_dir / "metrics"
        self.predictions_dir = self.save_dir / "predictions"
        
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Create segmentation visualizer
        self.segmentation_visualizer = SegmentationVisualizer(
            class_names=classes,  # Changed from 'classes' to 'class_names' to match SegmentationVisualizer
            save_dir=str(self.visualizations_dir)
        )
        
        self.logger.info(f"Visualization Manager initialized: {self.save_dir}")
        
    def plot_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        epoch: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ):
        """
        Generate plots for all available metrics.
        
        Args:
            metrics_history: Dictionary of metric histories
            epoch: Current epoch (for naming files)
            save: Whether to save the plots
            show: Whether to show the plots
        """
        if not metrics_history:
            self.logger.warning("No metrics history provided for plotting")
            return
            
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.experiment_name} Training Metrics", fontsize=16)
        
        # Plot loss
        if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
            ax = axes[0, 0]
            ax.plot(metrics_history['train_loss'], label='Train Loss')
            ax.plot(metrics_history['val_loss'], label='Validation Loss')
            ax.set_title('Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        # Plot Dice coefficient
        if 'train_dice' in metrics_history and 'val_dice' in metrics_history:
            ax = axes[0, 1]
            ax.plot(metrics_history['train_dice'], label='Train Dice')
            ax.plot(metrics_history['val_dice'], label='Validation Dice')
            ax.set_title('Dice Coefficient')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dice')
            ax.legend()
            ax.grid(True)
        
        # Plot Precision
        if 'train_precision' in metrics_history and 'val_precision' in metrics_history:
            ax = axes[1, 0]
            ax.plot(metrics_history['train_precision'], label='Train Precision')
            ax.plot(metrics_history['val_precision'], label='Validation Precision')
            ax.set_title('Precision')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Precision')
            ax.legend()
            ax.grid(True)
        
        # Plot IoU (Jaccard)
        if 'train_jaccard' in metrics_history and 'val_jaccard' in metrics_history:
            ax = axes[1, 1]
            ax.plot(metrics_history['train_jaccard'], label='Train Jaccard')
            ax.plot(metrics_history['val_jaccard'], label='Validation Jaccard')
            ax.set_title('Jaccard Index (IoU)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if requested
        if save:
            epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
            save_path = self.metrics_dir / f"metrics{epoch_str}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.logger.info(f"Saved metrics plot to {save_path}")
            
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
    def visualize_predictions(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        num_samples: int = 4,
        epoch: Optional[int] = None,
        stage: Optional[int] = None
    ):
        """
        Generate and save prediction visualizations.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for image data
            device: Device to run the model on
            num_samples: Number of samples to visualize
            epoch: Current epoch (for naming files)
            stage: Current stage (for naming files)
        """
        model.eval()
        
        # Get random indices
        indices = np.random.choice(
            len(dataloader.dataset),
            min(num_samples, len(dataloader.dataset)),
            replace=False
        )
        
        self.logger.info(f"Generating visualizations for {len(indices)} samples")
        
        # Generate visualizations
        for i, idx in enumerate(indices):
            # Get a sample
            img, target = dataloader.dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(img_batch)
                    
                # Process outputs based on model type
                if isinstance(outputs, list):
                    # Use the last stage output if the model returns multiple outputs
                    if model.get_predictions is not None:
                        pred = model.get_predictions(outputs)
                    else:
                        pred = torch.sigmoid(outputs[-1]) > 0.5
                else:
                    # Single output model
                    pred = torch.sigmoid(outputs) > 0.5
                    
            # Move to CPU and remove batch dimension
            pred = pred.cpu().squeeze(0)
            
            # Generate filename
            stage_str = f"_stage{stage}" if stage is not None else ""
            epoch_str = f"_epoch{epoch}" if epoch is not None else ""
            filename = f"sample{idx}{stage_str}{epoch_str}.png"
            save_path = self.predictions_dir / filename
            
            # Use the visualizer to save the visualization
            self.segmentation_visualizer.visualize_prediction(
                img,
                pred,
                target,
                show=False,
                save_path=str(save_path)
            )
            
        self.logger.info(f"Saved visualizations to {self.predictions_dir}")
        
    def generate_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None
    ):
        """
        Generate and save confusion matrix.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            epoch: Current epoch (for naming files)
        """
        epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
        save_path = self.visualizations_dir / f"confusion_matrix{epoch_str}.png"
        
        self.segmentation_visualizer.plot_confusion_matrix(
            predictions,
            targets,
            save_path=str(save_path)
        )
        
        self.logger.info(f"Saved confusion matrix to {save_path}")
        
    def log_epoch_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        checkpoint_saved: bool = False,
        patience_counter: int = 0,
        patience_limit: int = 10,
        learning_rate: float = 0.001
    ):
        """
        Log a formatted summary of epoch metrics.
        
        Args:
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            epoch: Current epoch
            checkpoint_saved: Whether a checkpoint was saved this epoch
            patience_counter: Current patience counter for early stopping
            patience_limit: Patience limit for early stopping
            learning_rate: Current learning rate
        """
        # Create ASCII table characters for better compatibility
        border_char = "="
        side_border = "|"
        corner_tl = "+"
        corner_tr = "+"
        corner_bl = "+"
        corner_br = "+"
        t_down = "+"
        t_up = "+"
        t_right = "+"
        t_left = "+"
        cross = "+"
        
        table_width = 110
        border_top = corner_tl + border_char * (table_width-2) + corner_tr
        border_bottom = corner_bl + border_char * (table_width-2) + corner_br
        border_mid = t_right + border_char * (table_width-2) + t_left
        
        # Create header with epoch info and checkpoint/early stopping status
        epoch_text = f"EPOCH {epoch:3d} SUMMARY"
        status_text = " [CHECKPOINT SAVED]" if checkpoint_saved else f" [EARLY STOPPING: {patience_counter}/{patience_limit}]"
        
        padding = (table_width - len(epoch_text) - len(status_text) - 2) // 2
        header_line = f"{side_border}{' ' * padding}{epoch_text}{' ' * (table_width - len(epoch_text) - len(status_text) - 2 - padding)}{status_text}{side_border}"
        
        # Build and log the header
        self.logger.info("\n" + border_top)
        self.logger.info(header_line)
        self.logger.info(border_mid)
        
        # Column headers with better alignment
        col_widths = [22, 25, 25, 34]
        header_format = f"{side_border} {{:<{col_widths[0]}}} {side_border} {{:<{col_widths[1]}}} {side_border} {{:<{col_widths[2]}}} {side_border} {{:<{col_widths[3]}}} {side_border}"
        row_format = f"{side_border} {{:<{col_widths[0]}}} {side_border} {{:{col_widths[1]}.5f}} {side_border} {{:{col_widths[2]}.5f}} {side_border} {{:<{col_widths[3]}}} {side_border}"
        
        # Print column headers
        header_row = header_format.format("Metric", "Training", "Validation", "Difference")
        self.logger.info(header_row)
        
        # Separator after header
        sep_line = t_right
        for i, width in enumerate(col_widths):
            sep_line += border_char * (width + 2)
            sep_line += cross if i < len(col_widths) - 1 else t_left
        self.logger.info(sep_line)
        
        # Define the order of metrics to display
        metric_order = [
            ('base_loss', 'Base Loss'),
            ('loss', 'Loss with Reward'),
            ('dice', 'Dice'),
            ('precision', 'Precision'),
            ('iou', 'Jaccard Index'),
            ('recall', 'Recall')
        ]
        
        # Log metrics in the specified order
        for metric_key, display_name in metric_order:
            if metric_key in train_metrics and metric_key in val_metrics:
                train_val = train_metrics[metric_key]
                val_val = val_metrics[metric_key]
                diff_val = train_val - val_val
                
                # For these metrics, higher validation is better (except losses)
                if metric_key in ['base_loss', 'loss']:
                    diff_indicator = "[BETTER]" if diff_val > 0 else "[WORSE]"
                else:
                    diff_indicator = "[BETTER]" if diff_val < 0 else "[WORSE]"
                
                diff_text = f"{diff_val:.5f} {diff_indicator}"
                self.logger.info(row_format.format(display_name, train_val, val_val, diff_text))
                
        # Add stage-specific metrics if available
        stage_metrics = {}
        
        # Find all available stage metrics
        for key in train_metrics.keys():
            if key.startswith('stage'):
                parts = key.split('_')
                if len(parts) >= 2:
                    stage = parts[0]  # e.g. 'stage1'
                    metric = '_'.join(parts[1:])  # e.g. 'dice'
                    
                    if stage not in stage_metrics:
                        stage_metrics[stage] = {}
                    
                    if metric in ['dice', 'precision', 'recall', 'iou']:
                        if metric not in stage_metrics[stage]:
                            train_val = train_metrics.get(key, 0)
                            val_val = val_metrics.get(key, 0)
                            diff_val = train_val - val_val
                            # For these metrics, higher validation is better
                            diff_indicator = "[BETTER]" if diff_val < 0 else "[WORSE]"
                            
                            stage_metrics[stage][metric] = {
                                'train': train_val,
                                'val': val_val,
                                'diff': diff_val,
                                'indicator': diff_indicator
                            }
        
        # Add stage metrics in a readable format
        if stage_metrics:
            # Add a separator before stage metrics
            self.logger.info(sep_line)
            
            for stage, metrics in sorted(stage_metrics.items()):
                stage_num = stage.replace('stage', 'Stage ')
                
                # Make a header for the stage
                stage_padding = (table_width - len(stage_num) - 2) // 2
                stage_header = f"{side_border}{' ' * stage_padding}{stage_num}{' ' * (table_width - len(stage_num) - 2 - stage_padding)}{side_border}"
                self.logger.info(stage_header)
                self.logger.info(sep_line)
                
                for metric_name in ['dice', 'iou', 'precision', 'recall']:
                    if metric_name in metrics:
                        values = metrics[metric_name]
                        # Capitalize the metric name for display
                        display_name = f"  {metric_name.capitalize()}"
                        # Format IoU as Jaccard Index
                        if metric_name == 'iou':
                            display_name = "  Jaccard Index"
                        
                        diff_text = f"{values['diff']:.5f} {values['indicator']}"
                        self.logger.info(row_format.format(
                            display_name, values['train'], values['val'], diff_text
                        ))
        
        # Bottom border of the table
        self.logger.info(border_bottom)
        
        # Log learning rate
        self.logger.info(f"Current learning rate: {learning_rate:.2e}\n")