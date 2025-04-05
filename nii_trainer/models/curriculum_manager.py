import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from ..configs.config import TrainingConfig

class CurriculumManager:
    """
    Manages curriculum learning for staged training.
    Controls which stages are active, learning rate scheduling,
    and freezing/unfreezing of model components.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize curriculum manager.
        
        Args:
            config: Training configuration
            model: Neural network model
            logger: Optional logger for logging curriculum events
        """
        self.config = config
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        # Default curriculum parameters
        self.curriculum_params = {
            'stage_schedule': [(0, config.epochs)],  # Default: train all stages for all epochs
            'learning_rates': [config.learning_rate],  # Default: use config learning rate for all stages
            'stage_freezing': [False]  # Default: don't freeze any stages
        }
        
        self.current_epoch = 0
        self.active_stages = [0]  # Start with just the first stage by default
        self.frozen_stages = []   # Keep track of which stages are frozen
        # Keep track of the current primary stage (for metrics display)
        self.current_primary_stage = 0
        
    def configure_curriculum(self, params: Dict):
        """
        Configure the curriculum with custom parameters.
        
        Args:
            params: Dictionary with curriculum parameters
        """
        if not params:
            self.logger.warning("No curriculum parameters provided, using defaults")
            return
            
        self.logger.info("Configuring curriculum learning")
        
        if 'stage_schedule' in params:
            self.curriculum_params['stage_schedule'] = params['stage_schedule']
            self.logger.info(f"Stage schedule: {params['stage_schedule']}")
            
        if 'learning_rates' in params:
            self.curriculum_params['learning_rates'] = params['learning_rates']
            self.logger.info(f"Learning rates: {params['learning_rates']}")
            
        if 'stage_freezing' in params:
            self.curriculum_params['stage_freezing'] = params['stage_freezing']
            self.logger.info(f"Stage freezing: {params['stage_freezing']}")
            
        # Start with just the first stage
        self.active_stages = [0]
        self.current_primary_stage = 0
        
    def update_active_stages(self, curriculum_stage: int) -> List[int]:
        """
        Update which stages are active based on curriculum stage.
        
        Args:
            curriculum_stage: Current curriculum stage index
            
        Returns:
            List of active stage indices
        """
        if 'stage_schedule' not in self.curriculum_params:
            # If no schedule provided, all stages are active
            num_stages = len(self.model.stages) if hasattr(self.model, 'stages') else 1
            self.active_stages = list(range(num_stages))
            return self.active_stages
            
        # Get the stage configuration from the schedule for this curriculum stage
        schedule = self.curriculum_params['stage_schedule']
        if curriculum_stage < len(schedule):
            stage_idx, _ = schedule[curriculum_stage]
            
            # Update the current primary stage (the one we're actually training)
            self.current_primary_stage = stage_idx
            
            # In strict curriculum mode, we only activate the current stage
            # This improves performance and reduces GPU memory usage
            self.active_stages = [stage_idx]
            
            self.logger.info(f"Curriculum stage {curriculum_stage+1}: Active model stage = {stage_idx+1}")
            
            return self.active_stages
        else:
            # If curriculum stage is out of range, activate all stages
            num_stages = len(self.model.stages) if hasattr(self.model, 'stages') else 1
            self.active_stages = list(range(num_stages))
            return self.active_stages
        
    def get_learning_rate(self) -> float:
        """
        Get current learning rate based on curriculum.
        
        Returns:
            Learning rate for current epoch
        """
        if not self.active_stages:
            return self.config.learning_rate
            
        # Get learning rate for current primary stage
        if 'learning_rates' not in self.curriculum_params:
            return self.config.learning_rate
            
        learning_rates = self.curriculum_params['learning_rates']
        
        # Use the learning rate for the current curriculum stage
        current_schedule_idx = 0
        for i, (stage_idx, _) in enumerate(self.curriculum_params['stage_schedule']):
            if stage_idx == self.current_primary_stage:
                current_schedule_idx = i
                break
                
        if current_schedule_idx < len(learning_rates):
            return learning_rates[current_schedule_idx]
        else:
            return learning_rates[-1]  # Use last defined learning rate
            
    def apply_stage_freezing(self):
        """Freeze all stages except the current primary stage.
        
        This implementation unconditionally freezes every stage
        other than the current primary stage.
        """
        # Reset the frozen stages list.
        self.frozen_stages = []
        
        # Freeze every stage except the current primary stage.
        for stage_idx, stage in enumerate(self.model.stages):
            if stage_idx == self.current_primary_stage:
                # Unfreeze the active stage.
                for param in stage.parameters():
                    param.requires_grad = True
                self.logger.info(f"Stage {stage_idx+1} active and unfrozen")
            else:
                # Freeze all other stages.
                for param in stage.parameters():
                    param.requires_grad = False
                self.logger.info(f"Stage {stage_idx+1} frozen")
                self.frozen_stages.append(stage_idx)
        
        self.logger.info(f"Active stage: {self.current_primary_stage+1}")
        if self.frozen_stages:
            self.logger.info(f"Frozen stages: {[s+1 for s in self.frozen_stages]}")
        
        return self.frozen_stages



    def get_active_and_frozen_stages(self) -> Tuple[List[int], List[int]]:
        """
        Returns both active and frozen stages.
        
        Returns:
            Tuple of (active stages indices, frozen stages indices)
        """
        return self.active_stages, self.frozen_stages
    
    def get_current_primary_stage(self) -> int:
        """
        Returns the current primary stage being trained.
        
        Returns:
            Index of the current primary stage
        """
        return self.current_primary_stage
        
    def are_stages_frozen(self) -> bool:
        """
        Check if any stages are currently frozen.
        
        Returns:
            Boolean indicating if any stages are frozen
        """
        return len(self.frozen_stages) > 0

    def format_metrics_for_logging(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for CSV-style logging in the desired order.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted metrics string for logging
        """
        # Desired metric order
        metric_keys = ['base_loss', 'loss', 'dice', 'precision', 'recall', 'iou']
        values = []
        
        # Format each metric value with 4 decimal places
        for key in metric_keys:
            if key in metrics:
                # Format the value with 4 decimal places
                value = metrics[key]
                if torch.is_tensor(value):
                    value = value.item()
                values.append(f"{value:.4f}")
            else:
                values.append("NA")
        
        # Join values with commas for CSV format
        return ",".join(values)
    
    def get_metrics_header(self) -> str:
        """
        Get header row for metrics CSV.
        
        Returns:
            Header string
        """
        return "base_loss,loss_w_r,dice,precision,recall,jaccard"