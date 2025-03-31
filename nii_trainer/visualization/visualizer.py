import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import seaborn as sns
from ..configs.config import ModelConfig

class SegmentationVisualizer:
    """
    Visualization tools for segmentation predictions.
    Supports:
    - Multi-class visualization with customizable colormaps
    - Side-by-side comparison of predictions vs ground truth
    - Overlay views
    - Uncertainty visualization
    - Metric plots and training curves
    """
    def __init__(self, class_names: List[str], save_dir: Optional[str] = None):
        self.class_names = class_names
        self.save_dir = Path(save_dir) if save_dir else None
        self.class_colors = self._generate_colors(len(class_names))
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors for each class."""
        colors = [(0, 0, 0)]  # Background is black
        if num_classes > 1:
            # Generate distinct colors using HSV color space
            hsv_colors = [(x/float(num_classes-1), 0.8, 0.8) for x in range(num_classes-1)]
            import colorsys
            rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
            colors.extend(rgb_colors)
        return colors
    
    def visualize_prediction(
        self,
        image: torch.Tensor,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize a single prediction with optional ground truth and uncertainty.
        Args:
            image: Input image [1, H, W]
            prediction: Predicted segmentation [H, W]
            target: Optional ground truth [H, W]
            uncertainty: Optional uncertainty map [H, W]
            show: Whether to display the plot
            save_path: Optional path to save the visualization
        """
        # Convert tensors to numpy
        image = image.squeeze().cpu().numpy()
        prediction = prediction.squeeze().cpu().numpy()
        if target is not None:
            target = target.squeeze().cpu().numpy()
        if uncertainty is not None:
            uncertainty = uncertainty.squeeze().cpu().numpy()
            
        num_plots = 2 if target is None else 3
        if uncertainty is not None:
            num_plots += 1
            
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        if num_plots == 1:
            axes = [axes]
            
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Create segmentation overlay
        seg_img = np.zeros((*image.shape, 3))
        for i, color in enumerate(self.class_colors):
            mask = prediction == i
            for c in range(3):
                seg_img[..., c][mask] = color[c]
                
        # Plot prediction
        axes[1].imshow(image, cmap='gray')
        axes[1].imshow(seg_img, alpha=0.5)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Plot ground truth if available
        if target is not None:
            gt_img = np.zeros((*image.shape, 3))
            for i, color in enumerate(self.class_colors):
                mask = target == i
                for c in range(3):
                    gt_img[..., c][mask] = color[c]
                    
            axes[2].imshow(image, cmap='gray')
            axes[2].imshow(gt_img, alpha=0.5)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
        # Plot uncertainty if available
        if uncertainty is not None:
            uncertainty_idx = 2 if target is None else 3
            axes[uncertainty_idx].imshow(uncertainty, cmap='hot')
            axes[uncertainty_idx].set_title('Uncertainty')
            axes[uncertainty_idx].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training metrics history."""
        num_metrics = len(metrics)
        fig, axes = plt.subplots(
            (num_metrics + 1) // 2, 2,
            figsize=(15, 5 * ((num_metrics + 1) // 2))
        )
        axes = axes.flatten()
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', label=metric_name)
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
    def visualize_batch(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        max_samples: int = 8,
        save_dir: Optional[str] = None
    ) -> None:
        """Visualize a batch of predictions."""
        batch_size = min(images.shape[0], max_samples)
        for i in range(batch_size):
            save_path = None
            if save_dir:
                save_path = str(Path(save_dir) / f"sample_{i}.png")
                
            self.visualize_prediction(
                images[i],
                predictions[i],
                targets[i] if targets is not None else None,
                show=(save_path is None),
                save_path=save_path
            )
            
    def plot_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix for segmentation results."""
        num_classes = len(self.class_names)
        conf_matrix = torch.zeros(num_classes, num_classes)
        
        for c1 in range(num_classes):
            for c2 in range(num_classes):
                conf_matrix[c1, c2] = torch.sum(
                    (predictions == c1) & (targets == c2)
                ).item()
                
        # Normalize by row
        conf_matrix = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt='.2f',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Predicted')
        plt.xlabel('True')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()