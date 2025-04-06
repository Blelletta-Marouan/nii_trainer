"""Early stopping handler for training monitoring."""
import logging
from typing import Optional

class EarlyStoppingHandler:
    """Handles early stopping logic during training."""
    
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize early stopping handler.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            logger: Optional logger for status updates
        """
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best loss: {self.best_loss:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True
            return False
            
    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        
    @property
    def should_save_checkpoint(self) -> bool:
        """Check if current model is best so far."""
        return self.counter == 0