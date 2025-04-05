import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any
import logging

class GradientManager:
    """
    Manages gradient accumulation, updating, and optimization steps.
    Handles mixed precision training and gradient scaling.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 1,
        use_mixed_precision: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the gradient manager.
        
        Args:
            optimizer: PyTorch optimizer
            accumulation_steps: Number of batches to accumulate gradients over
            use_mixed_precision: Whether to use mixed precision training
            device: Device to use for training
            logger: Optional logger for logging information
        """
        self.optimizer = optimizer
        self.accumulation_steps = max(1, accumulation_steps)  # Ensure at least 1
        self.use_mixed_precision = use_mixed_precision
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize scaler for mixed precision if needed
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Internal counters
        self.current_step = 0
        self.batch_in_accumulation = 0
        
        # Log configuration
        self.logger.info(f"Gradient Manager initialized with accumulation steps: {accumulation_steps}")
        self.logger.info(f"Mixed precision training: {use_mixed_precision}")
        
    def zero_grad(self):
        """
        Zero gradients when starting a new accumulation cycle.
        Should be called before the first batch in an accumulation cycle.
        """
        if self.batch_in_accumulation == 0:
            self.optimizer.zero_grad()
            
    def backward_and_step(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients and perform optimizer step when needed.
        
        Args:
            loss: Loss tensor to backpropagate
            
        Returns:
            True if optimizer step was performed, False otherwise
        """
        # Normalize the loss by accumulation steps to keep gradients comparable
        normalized_loss = loss / self.accumulation_steps
        
        # Backward pass with appropriate scaling
        if self.use_mixed_precision:
            self.scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()
            
        # Update counter
        self.batch_in_accumulation += 1
        
        # Check if we should perform optimizer step
        if self.batch_in_accumulation >= self.accumulation_steps:
            self._perform_optimizer_step()
            self.batch_in_accumulation = 0
            self.current_step += 1
            return True
        
        return False
    
    def _perform_optimizer_step(self):
        """
        Perform optimizer step with gradient scaling if using mixed precision.
        """
        if self.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
    def get_lr(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
        
    def set_lr(self, lr: float):
        """
        Set learning rate for all parameter groups.
        
        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr