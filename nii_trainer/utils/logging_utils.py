import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    experiment_name: str,
    save_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for the training pipeline.
    Args:
        experiment_name: Name of the experiment for the log file
        save_dir: Directory to save log file. If None, only console logging is setup
        level: Logging level
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler with detailed format if save_dir is provided
    if save_dir:
        log_dir = Path(save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
    return logger

def log_config(logger: logging.Logger, config: dict) -> None:
    """Log configuration parameters."""
    logger.info("=== Configuration ===")
    for section, params in config.items():
        logger.info(f"\n[{section}]")
        for key, value in params.items():
            logger.info(f"{key}: {value}")
    logger.info("===================")