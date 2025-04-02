from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import numpy as np
import logging

from ..configs.config import TrainerConfig
from ..visualization.visualizer import SegmentationVisualizer
from ..losses.losses import CascadedLossWithReward
from .cascaded_unet import FlexibleCascadedUNet

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
        model: FlexibleCascadedUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize loss function
        self.criterion = CascadedLossWithReward(config.loss)
        
        # Training utilities
        if config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup directories
        self.exp_dir = Path(config.save_dir) / config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = SegmentationVisualizer(
            config.data.classes,
            save_dir=str(self.exp_dir / 'visualizations')
        )
        
        # Metrics history
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_base_loss': [], 'val_base_loss': [],  # Add base loss tracking
            'train_metrics': [], 'val_metrics': []
        }
        
        self.logger.info(f"Trainer initialized with {type(model).__name__}")
        self.logger.info(f"Using device: {config.training.device}")
        self.logger.info(f"Mixed precision training: {config.training.mixed_precision}")
        
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
            self.logger.info(f"New best model found at epoch {self.current_epoch}")
            torch.save(
                checkpoint,
                self.exp_dir / 'best_model.pth'
            )
            
        # Save metrics history
        with open(self.exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.training.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics_history = checkpoint['metrics_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            
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
                with autocast('cuda', enabled=self.config.training.mixed_precision):
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
                
                # Update progress bar with comprehensive metrics
                avg_loss = total_loss / (batch_idx + 1)
                
                # Track base loss separately if available
                base_loss = metrics_dict.get('base_loss', loss)
                
                # Extract individual metrics for display
                postfix_dict = {
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                }
                
                # Add base loss to display if available
                if 'base_loss' in metrics_dict:
                    postfix_dict['base_loss'] = f'{metrics_dict["base_loss"].item():.4f}'
                
                # Add per-stage metrics for each available stage
                stages_count = len(self.config.cascade.stages)
                for stage in range(stages_count):
                    stage_key = f'stage{stage+1}'
                    # Add key metrics for this stage if available
                    if f'{stage_key}_dice' in metrics_dict:
                        postfix_dict[f'S{stage+1}_dice'] = f'{metrics_dict[f"{stage_key}_dice"].item():.3f}'
                    if f'{stage_key}_precision' in metrics_dict:
                        postfix_dict[f'S{stage+1}_prec'] = f'{metrics_dict[f"{stage_key}_precision"].item():.3f}'
                    if f'{stage_key}_recall' in metrics_dict:
                        postfix_dict[f'S{stage+1}_rec'] = f'{metrics_dict[f"{stage_key}_recall"].item():.3f}'
                    if f'{stage_key}_iou' in metrics_dict:
                        postfix_dict[f'S{stage+1}_iou'] = f'{metrics_dict[f"{stage_key}_iou"].item():.3f}'
                
                pbar.set_postfix(postfix_dict)
                
                # Log every 100 batches
                if batch_idx % 100 == 0:
                    self.logger.debug(
                        f"Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}"
                    )
                
        # Calculate average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = total_loss / len(self.train_loader)
        
        # Calculate average base loss if available
        if 'base_loss' in all_metrics[0]:
            base_loss_sum = sum(m['base_loss'].item() for m in all_metrics)
            avg_metrics['base_loss'] = base_loss_sum / len(self.train_loader)
        
        # Store in metrics history
        self.metrics_history['train_loss'].append(avg_metrics['loss'])
        if 'base_loss' in avg_metrics:
            self.metrics_history['train_base_loss'].append(avg_metrics['base_loss'])
        
        # Log epoch metrics
        self.logger.info(
            f"Epoch {self.current_epoch} - Training Loss: {avg_metrics['loss']:.4f}"
        )
        if 'base_loss' in avg_metrics:
            self.logger.info(f"Epoch {self.current_epoch} - Training Base Loss: {avg_metrics['base_loss']:.4f}")
            
        for key, value in avg_metrics.items():
            if key not in ['loss', 'base_loss']:
                self.logger.debug(f"Training {key}: {value:.4f}")
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        all_predictions = []
        all_targets = []
        
        self.logger.info("Starting validation...")
        
        # Create progress bar with detailed metrics
        val_pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch_idx, (images, targets) in enumerate(val_pbar):
            images = images.to(self.config.training.device)
            targets = targets.to(self.config.training.device)
            
            # Forward pass
            with autocast('cuda', enabled=self.config.training.mixed_precision):
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
            
            # Update progress bar with current and average metrics
            avg_loss = total_loss / (batch_idx + 1)
            
            # Extract individual metrics for display
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            }
            
            # Add base loss to display if available
            if 'base_loss' in metrics_dict:
                postfix_dict['base_loss'] = f'{metrics_dict["base_loss"].item():.4f}'
            
            # Add per-stage metrics for each available stage
            stages_count = len(self.config.cascade.stages)
            for stage in range(stages_count):
                stage_key = f'stage{stage+1}'
                # Add key metrics for this stage if available
                if f'{stage_key}_dice' in metrics_dict:
                    postfix_dict[f'S{stage+1}_dice'] = f'{metrics_dict[f"{stage_key}_dice"].item():.3f}'
                if f'{stage_key}_precision' in metrics_dict:
                    postfix_dict[f'S{stage+1}_prec'] = f'{metrics_dict[f"{stage_key}_precision"].item():.3f}'
                if f'{stage_key}_recall' in metrics_dict:
                    postfix_dict[f'S{stage+1}_rec'] = f'{metrics_dict[f"{stage_key}_recall"].item():.3f}'
                if f'{stage_key}_iou' in metrics_dict:
                    postfix_dict[f'S{stage+1}_iou'] = f'{metrics_dict[f"{stage_key}_iou"].item():.3f}'
            
            val_pbar.set_postfix(postfix_dict)
            
        # Calculate average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Calculate average base loss if available
        if 'base_loss' in all_metrics[0]:
            base_loss_sum = sum(m['base_loss'].item() for m in all_metrics)
            avg_metrics['base_loss'] = base_loss_sum / len(self.val_loader)
        
        # Store in metrics history
        self.metrics_history['val_loss'].append(avg_metrics['loss'])
        if 'base_loss' in avg_metrics:
            self.metrics_history['val_base_loss'].append(avg_metrics['base_loss'])
        
        # Generate validation visualizations
        if self.current_epoch % 5 == 0:  # Visualize every 5 epochs
            self.logger.info("Generating validation visualizations...")
            self._generate_validation_visualizations(
                torch.cat(all_predictions),
                torch.cat(all_targets)
            )
            
        return avg_metrics, avg_metrics['loss']
    
    def trainer(
        self,
        stage_schedule: List[Tuple[int, int]],
        learning_rates: Optional[List[float]] = None,
        stage_freezing: Optional[List[bool]] = None
    ) -> None:
        """
        Train the model with curriculum learning, focusing on different stages over time.
        
        Args:
            stage_schedule: List of (stage_idx, num_epochs) tuples
            learning_rates: Optional list of learning rates for each stage
            stage_freezing: Optional list of booleans indicating whether to freeze previous stages
        """
        self.logger.info("Starting training with curriculum learning...")
        
        # Default learning rates if not provided
        if learning_rates is None:
            learning_rates = [self.config.training.learning_rate] * len(stage_schedule)
            
        # Default freezing behavior if not provided
        if stage_freezing is None:
            stage_freezing = [False] * len(stage_schedule)
            
        # Keep track of best metrics for early stopping
        best_metrics = {
            "loss": float("inf"),
            "dice": 0.0
        }
        patience_counter = 0
        
        # Robust training approach: train one stage at a time directly
        for stage_idx, (stage, num_epochs) in enumerate(stage_schedule):
            stage_name = f"Stage {stage + 1}"
            self.logger.info(f"Training {stage_name} for {num_epochs} epochs")
            
            # Set learning rate for this stage
            for param_group in self.optimizer.param_groups:
                param_group['learning_rate'] = learning_rates[stage_idx]
                
            # Freeze previous stages if requested
            if stage_idx > 0 and stage_freezing[stage_idx]:
                self._freeze_stages(range(stage))
                
            # Get direct access to the stage
            current_stage = self.model.stages[stage]
            
            # Train the stage directly
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Create progress bar for this epoch
                train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} | {stage_name}")
                
                # Training loop
                train_metrics = {}
                total_loss = 0.0
                total_dice = 0.0
                total_precision = 0.0
                total_recall = 0.0
                train_batches = 0
                
                for batch_idx, (images, targets) in enumerate(train_pbar):
                    # Move to device
                    images = images.to(self.config.training.device)
                    targets = targets.to(self.config.training.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass - directly use stage
                    with autocast('cuda', enabled=self.config.training.mixed_precision):
                        # For first stage, use images directly
                        if stage == 0:
                            stage_input = images
                        else:
                            # For second stage, create binary liver mask first
                            with torch.no_grad():
                                liver_output = self.model.stages[0](images)
                                liver_mask = (torch.sigmoid(liver_output) > 0.5).float()
                            # Concatenate with original image
                            stage_input = torch.cat([images, liver_mask], dim=1)
                        
                        # Get output from current stage
                        stage_output = current_stage(stage_input)
                        
                        # Compute loss - get appropriate target for this stage
                        if stage == 0:  # Liver stage: anything non-background is liver
                            stage_target = (targets > 0).float()
                        else:  # Tumor stage: class 2 is tumor, within liver region
                            stage_target = (targets == 2).float()
                            

                        # Compute loss
                        loss = F.binary_cross_entropy_with_logits(
                            stage_output, 
                            stage_target.unsqueeze(1)
                        )
                    
                    # Backward pass and optimizer step
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                        
                    # Calculate additional metrics
                    with torch.no_grad():
                        # Convert predictions to binary masks
                        pred = (torch.sigmoid(stage_output) > 0.5).float()
                        
                        # Dice coefficient
                        dice = self._dice_score(pred, stage_target.unsqueeze(1))
                        
                        # Precision
                        true_positives = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 1))
                        false_positives = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 0))
                        precision = true_positives / (true_positives + false_positives + 1e-7)
                        
                        # Recall
                        false_negatives = torch.sum((pred == 0) & (stage_target.unsqueeze(1) == 1))
                        recall = true_positives / (true_positives + false_negatives + 1e-7)
                        
                        # Jaccard index (IoU)
                        intersection = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 1))
                        union = torch.sum((pred == 1) | (stage_target.unsqueeze(1) == 1))
                        iou = intersection / (union + 1e-7)
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_dice += dice.item()
                    total_precision += precision.item()
                    total_recall += recall.item()
                    total_iou = iou.item() if 'total_iou' in locals() else iou.item()
                    train_batches += 1
                    
                    # Update progress bar with comprehensive metrics
                    train_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                        "dice": f"{dice.item():.4f}",
                        "prec": f"{precision.item():.4f}",
                        "rec": f"{recall.item():.4f}",
                        "iou": f"{iou.item():.4f}"
                    })
                
                # Compute average training metrics
                train_metrics["loss"] = total_loss / train_batches
                train_metrics["dice"] = total_dice / train_batches
                train_metrics["precision"] = total_precision / train_batches
                train_metrics["recall"] = total_recall / train_batches
                train_metrics["iou"] = total_iou / train_batches
                
                # Validation
                val_metrics = self._validate_stage(current_stage, stage)
                
                # Check early stopping
                improved = False
                if val_metrics["loss"] < best_metrics["loss"]:
                    best_metrics["loss"] = val_metrics["loss"]
                    improved = True
                
                # Always log epoch summary after each epoch
                self._log_epoch_summary(train_metrics, val_metrics, improved)
    
                if improved:
                    patience_counter = 0
                    # Save checkpoint
                    save_path = f"checkpoints/{stage_name.lower().replace(' ', '_')}_best.pth"
                    self._save_checkpoint(save_path, current_stage, stage_idx)
                    self.logger.info(f"Saved new best model checkpoint to {save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training.patience:
                        self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                        break
                
                # Update the patience counter in class so it's available to _log_epoch_summary
                self.patience_counter = patience_counter
        
            # Unfreeze all stages for next round
            self._unfreeze_stages()
            
    def _validate_stage(self, stage_model, stage_idx):
        """Validate a specific stage."""
        stage_model.eval()
        
        val_loss = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_batches = 0
        
        # Create validation progress bar for stage-specific validation
        val_pbar = tqdm(self.val_loader, desc=f"Validating Stage {stage_idx+1}")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_pbar):
                # Move to device
                images = images.to(self.config.training.device)
                targets = targets.to(self.config.training.device)
                
                # For first stage, use images directly
                if stage_idx == 0:
                    stage_input = images
                    stage_target = (targets > 0).float()  # Liver is anything non-background
                else:
                    # For second stage, create binary liver mask first
                    with torch.no_grad():
                        liver_output = self.model.stages[0](images)
                        liver_mask = (torch.sigmoid(liver_output) > 0.5).float()
                    # Concatenate with original image
                    stage_input = torch.cat([images, liver_mask], dim=1)
                    stage_target = (targets == 2).float()  # Class 2 is tumor
                
                # Forward pass
                stage_output = stage_model(stage_input)
                
                # Compute loss
                loss = F.binary_cross_entropy_with_logits(
                    stage_output, 
                    stage_target.unsqueeze(1)
                )
                
                # Compute metrics
                pred = (torch.sigmoid(stage_output) > 0.5).float()
                dice = self._dice_score(pred, stage_target.unsqueeze(1))
                
                # Precision
                true_positives = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 1))
                false_positives = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 0))
                precision = true_positives / (true_positives + false_positives + 1e-7)
                
                # Recall
                false_negatives = torch.sum((pred == 0) & (stage_target.unsqueeze(1) == 1))
                recall = true_positives / (true_positives + false_negatives + 1e-7)
                
                # Jaccard index (IoU)
                intersection = torch.sum((pred == 1) & (stage_target.unsqueeze(1) == 1))
                union = torch.sum((pred == 1) | (stage_target.unsqueeze(1) == 1))
                iou = intersection / (union + 1e-7)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()
                val_precision += precision.item()
                val_recall += recall.item()
                val_iou = iou.item() if 'val_iou' in locals() else iou.item()
                val_batches += 1
                
                # Update progress bar with current and average metrics
                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{val_loss / (batch_idx + 1):.4f}",
                    "dice": f"{dice.item():.4f}",
                    "precision": f"{precision.item():.4f}",
                    "recall": f"{recall.item():.4f}",
                    "iou": f"{iou.item():.4f}"
                })
                
        # Compute average validation metrics
        metrics = {
            "loss": val_loss / val_batches,
            "dice": val_dice / val_batches,
            "precision": val_precision / val_batches,
            "recall": val_recall / val_batches,
            "iou": val_iou / val_batches
        }
        
        stage_model.train()
        return metrics
        
    def _dice_score(self, pred, target):
        """Compute Dice score."""
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        return (2.0 * intersection + smooth) / (union + smooth)
        
    def _freeze_stages(self, stage_indices):
        """Freeze specified stages."""
        for idx in stage_indices:
            for param in self.model.stages[idx].parameters():
                param.requires_grad = False
                
    def _unfreeze_stages(self):
        """Unfreeze all stages."""
        for stage in self.model.stages:
            for param in stage.parameters():
                param.requires_grad = True
                
    def _save_checkpoint(self, path, stage_model, stage_idx):
        """Save a checkpoint for a specific stage."""
        checkpoint = {
            "stage_idx": stage_idx,
            "state_dict": stage_model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    
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
        
    def _log_epoch_summary(
        self,
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float],
        checkpoint_saved: bool
    ) -> None:
        """
        Log a formatted table summary of epoch metrics.
        
        Args:
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            checkpoint_saved: Whether a checkpoint was saved this epoch
        """
        # Create header for the summary table with ASCII characters instead of Unicode
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
        epoch_text = f"EPOCH {self.current_epoch:3d} SUMMARY"
        status_text = " [CHECKPOINT SAVED]" if checkpoint_saved else f" [EARLY STOPPING: {self.patience_counter}/{self.config.training.patience}]"
        
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
        
        # Add loss metrics first
        train_loss = train_metrics['loss']
        val_loss = val_metrics['loss']
        diff_loss = train_loss - val_loss
        diff_indicator = "[WORSE]" if diff_loss < 0 else "[BETTER]"  # Use text instead of emoji
        
        diff_text = f"{diff_loss:.5f} {diff_indicator}"
        self.logger.info(row_format.format("Loss", train_loss, val_loss, diff_text))
        
        if 'base_loss' in train_metrics and 'base_loss' in val_metrics:
            train_base = train_metrics['base_loss']
            val_base = val_metrics['base_loss']
            diff_base = train_base - val_base
            diff_indicator = "[WORSE]" if diff_base < 0 else "[BETTER]"
            
            diff_text = f"{diff_base:.5f} {diff_indicator}"
            self.logger.info(row_format.format("Base Loss", train_base, val_base, diff_text))
        
        # Add a separator before performance metrics
        self.logger.info(sep_line)
        
        # Add common direct metrics in a good grouping
        for metric_name in ['dice', 'iou', 'precision', 'recall']:
            if metric_name in train_metrics and metric_name in val_metrics:
                train_val = train_metrics[metric_name]
                val_val = val_metrics[metric_name]
                diff_val = train_val - val_val
                # For these metrics, higher validation is better 
                diff_indicator = "[BETTER]" if diff_val < 0 else "[WORSE]"
                
                # Format the metric name nicely
                display_name = metric_name.capitalize()
                if metric_name == 'iou':
                    display_name = "Jaccard Index"
                
                diff_text = f"{diff_val:.5f} {diff_indicator}"
                self.logger.info(row_format.format(display_name, train_val, val_val, diff_text))
        
        # Add stage-specific metrics
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
        
        # Plot metrics if it's a save checkpoint epoch
        if checkpoint_saved and hasattr(self, 'visualizer'):
            metrics_plot_path = str(self.exp_dir / f'metrics_epoch_{self.current_epoch}.png')
            self.visualizer.plot_metrics(self.metrics_history, save_path=metrics_plot_path)
            self.logger.info(f"Metrics plot saved to {metrics_plot_path}")
            
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Current learning rate: {current_lr:.2e}\n")