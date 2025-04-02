from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import torch
from torch.utils.data import DataLoader

from ..configs.config import DataConfig
from ..data import MultiClassSegDataset, PairedTransform

def create_data_transforms(
    img_size: Tuple[int, int] = (512, 512),
    augment_train: bool = True,
    augment_val: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, PairedTransform]:
    """
    Create data transformations for training and validation datasets.
    
    Args:
        img_size: Target image size (width, height)
        augment_train: Whether to apply augmentation to training data
        augment_val: Whether to apply augmentation to validation data
        logger: Logger for logging information
        
    Returns:
        Dictionary containing train and val transformations
    """
    if logger:
        logger.info("Creating data transformations...")
        logger.info(f"  Target image size: {img_size}")
        logger.info(f"  Training augmentation: {augment_train}")
        logger.info(f"  Validation augmentation: {augment_val}")
    
    transform_train = PairedTransform(
        img_size=img_size,
        augment=augment_train
    )
    
    transform_val = PairedTransform(
        img_size=img_size,
        augment=augment_val
    )
    
    return {
        "train": transform_train,
        "val": transform_val
    }

def create_datasets(
    data_dir: Union[str, Path],
    class_map: Dict[str, int],
    transforms: Dict[str, PairedTransform],
    balance_train: bool = False,
    balance_val: bool = False,
    logger: Optional[logging.Logger] = None,
    experiment_name: str = "dataset"
) -> Dict[str, MultiClassSegDataset]:
    """
    Create training and validation datasets.
    
    Args:
        data_dir: Base directory containing processed data
        class_map: Mapping from class names to indices
        transforms: Dictionary of transformations
        balance_train: Whether to balance classes in training data
        balance_val: Whether to balance classes in validation data
        logger: Logger for logging information
        experiment_name: Name prefix for dataset loggers
        
    Returns:
        Dictionary containing train and val datasets
    """
    data_dir = Path(data_dir)
    
    if logger:
        logger.info("Creating datasets from processed data...")
    
    # Record loading time for performance analysis
    start_time = time.time()
    
    # Create training dataset
    train_dataset = MultiClassSegDataset(
        data_dir=str(data_dir / "train"),
        class_map=class_map,
        transform=transforms["train"],
        balance=balance_train,
        logger=logger,
        experiment_name=f"{experiment_name}_train"
    )
    
    # Create validation dataset if it exists
    val_dataset = None
    if (data_dir / "val").exists():
        val_dataset = MultiClassSegDataset(
            data_dir=str(data_dir / "val"),
            class_map=class_map,
            transform=transforms["val"],
            balance=balance_val,
            logger=logger,
            experiment_name=f"{experiment_name}_val"
        )
    
    # Create test dataset if it exists
    test_dataset = None
    if (data_dir / "test").exists():
        test_dataset = MultiClassSegDataset(
            data_dir=str(data_dir / "test"),
            class_map=class_map,
            transform=transforms["val"],  # Use validation transform for test
            balance=False,  # Never balance test data
            logger=logger,
            experiment_name=f"{experiment_name}_test"
        )
    
    # Log dataset creation time and sizes
    dataset_loading_time = time.time() - start_time
    
    if logger:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        if test_dataset:
            logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info(f"Dataset loading completed in {dataset_loading_time:.2f} seconds")
    
    # Log class distribution statistics for training data
    if logger and hasattr(train_dataset, 'class_presence'):
        logger.info("Class distribution in training dataset:")
        for cls_idx, cls_name in class_map.items():
            if cls_idx in train_dataset.class_presence:
                count = len(train_dataset.class_presence[cls_idx])
                percentage = (count / len(train_dataset)) * 100
                logger.info(f"  {cls_name}: {count} samples ({percentage:.1f}%)")
    
    datasets = {"train": train_dataset}
    if val_dataset:
        datasets["val"] = val_dataset
    if test_dataset:
        datasets["test"] = test_dataset
        
    return datasets

def create_dataloaders(
    datasets: Dict[str, MultiClassSegDataset],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, DataLoader]:
    """
    Create dataloaders from datasets.
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        logger: Logger for logging information
        
    Returns:
        Dictionary containing dataloaders
    """
    if logger:
        logger.info("Creating dataloaders...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Number of workers: {num_workers}")
        logger.info(f"  Pin memory: {pin_memory}")
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    loaders = {"train": train_loader}
    
    # Create validation dataloader if dataset exists
    if "val" in datasets:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        loaders["val"] = val_loader
    
    # Create test dataloader if dataset exists
    if "test" in datasets:
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        loaders["test"] = test_loader
    
    return loaders

def setup_data_pipeline(
    config: DataConfig,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, MultiClassSegDataset], Dict[str, DataLoader]]:
    """
    Set up complete data pipeline from configuration.
    
    Args:
        config: Data configuration
        logger: Logger for logging information
        
    Returns:
        Tuple of (datasets, dataloaders)
    """
    # Create data transformations
    transforms = create_data_transforms(
        img_size=config.img_size,
        augment_train=True,
        augment_val=False,
        logger=logger
    )
    
    # Create datasets
    datasets = create_datasets(
        data_dir=config.output_dir,
        class_map=config.class_map,
        transforms=transforms,
        balance_train=config.balance_dataset,
        balance_val=False,
        logger=logger,
        experiment_name=config.experiment_name if hasattr(config, 'experiment_name') else "dataset"
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        datasets=datasets,
        batch_size=config.batch_size,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
        pin_memory=True,
        logger=logger
    )
    
    return datasets, dataloaders