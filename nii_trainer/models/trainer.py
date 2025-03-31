from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import numpy as np

from ..configs.config import TrainerConfig
from ..visualization.visualizer import SegmentationVisualizer
from ..losses.losses import CascadedLossWithReward
from .cascaded_unet import CascadedUNet

class ModelTrainer:
    """
    Handles model training, evaluation, and inference with support for:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Metric tracking and visualization
    - Model checkpointing
    """
    def __init__(
        self,
        config: TrainerConfig,
        model: CascadedUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.config = config
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Initialize loss function
        self.criterion = CascadedLossWithReward(config.loss)
        
        # Training utilities
        self.scaler = GradScaler() if config.training.mixed_precision else None
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup directories
        self.exp_dir = Path(config.save_dir) / config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = SegmentationVisualizer(
            config.model.classes,
            save_dir=str(self.exp_dir / 'visualizations')
        )
        
        # Metrics history
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }
        
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        torch.save(
            checkpoint,
            self.exp_dir / 'latest_checkpoint.pth'
        )
        
        # Save best model if needed
        if is_best:
            torch.save(
                checkpoint,
                self.exp_dir / 'best_model.pth'
            )
            
        # Save metrics history
        with open(self.exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.training.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics_history = checkpoint['metrics_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_metrics = []
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.config.training.device)
                targets = targets.to(self.config.training.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                with autocast(enabled=self.config.training.mixed_precision):
                    predictions = self.model(images)
                    metrics_dict = self.criterion(predictions, targets)
                    loss = metrics_dict['total_loss']
                    
                # Backward pass with gradient scaling if mixed precision is enabled
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                    
                # Update metrics
                total_loss += loss.item()
                all_metrics.append(metrics_dict)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
        # Calculate average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = total_loss / len(self.train_loader)
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        all_predictions = []
        all_targets = []
        
        for images, targets in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.config.training.device)
            targets = targets.to(self.config.training.device)
            
            # Forward pass
            with autocast(enabled=self.config.training.mixed_precision):
                predictions = self.model(images)
                metrics_dict = self.criterion(predictions, targets)
                loss = metrics_dict['total_loss']
                
            # Store results
            total_loss += loss.item()
            all_metrics.append(metrics_dict)
            
            # Get final predictions
            final_pred = self.model.get_predictions(
                predictions,
                self.config.loss.threshold_per_class
            )
            all_predictions.append(final_pred.cpu())
            all_targets.append(targets.cpu())
            
        # Calculate average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Generate validation visualizations
        if self.current_epoch % 5 == 0:  # Visualize every 5 epochs
            self._generate_validation_visualizations(
                torch.cat(all_predictions),
                torch.cat(all_targets)
            )
            
        return avg_metrics, avg_metrics['loss']
    
    def train(self) -> None:
        """Main training loop."""
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics, val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                
            # Save metrics
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_metrics'].append(train_metrics)
            self.metrics_history['val_metrics'].append(val_metrics)
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Plot metrics
            if epoch % 5 == 0:
                self.visualizer.plot_metrics(
                    {
                        'Train Loss': self.metrics_history['train_loss'],
                        'Val Loss': self.metrics_history['val_loss']
                    },
                    save_path=str(self.exp_dir / f'metrics_epoch_{epoch}.png')
                )
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
                
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average a list of metric dictionaries."""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key].item() if torch.is_tensor(m[key]) 
                                      else m[key] for m in metrics_list])
        return avg_metrics
    
    def _generate_validation_visualizations(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 4
    ) -> None:
        """Generate and save validation visualizations."""
        indices = np.random.choice(
            len(predictions),
            min(num_samples, len(predictions)),
            replace=False
        )
        
        for idx in indices:
            # Get a random validation sample
            img = self.val_loader.dataset[idx][0]
            pred = predictions[idx]
            target = targets[idx]
            
            # Generate visualization
            save_path = str(
                self.exp_dir / 'visualizations' / 
                f'val_sample_{idx}_epoch_{self.current_epoch}.png'
            )
            
            self.visualizer.visualize_prediction(
                img,
                pred,
                target,
                show=False,
                save_path=save_path
            )
            
        # Generate confusion matrix
        self.visualizer.plot_confusion_matrix(
            predictions,
            targets,
            save_path=str(
                self.exp_dir / 'visualizations' /
                f'confusion_matrix_epoch_{self.current_epoch}.png'
            )
        )