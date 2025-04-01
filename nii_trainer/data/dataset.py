from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Dict
import random
from torchvision import transforms
from ..configs.config import DataConfig
from ..utils.logging_utils import setup_logger
import logging
class MultiClassSegDataset(Dataset):
    """
    A flexible dataset for multi-class segmentation that supports:
    - Configurable number of classes
    - Dataset balancing
    - Custom transforms
    - Different sampling strategies
    """
    def __init__(
        self,
        data_dir: str,
        class_map: Dict[int, str],
        transform=None,
        balance: bool = False,
        required_classes: Optional[List[int]] = None,
        logger: Optional[logging.Logger] = None,
        experiment_name: str = "dataset"
    ):
        # Setup logger if not provided
        if logger is None:
            logger = setup_logger(experiment_name, save_dir=data_dir)
        self.logger = logger

        self.data_dir = Path(data_dir)
        self.class_map = class_map
        self.transform = transform
        self.balance = balance
        self.required_classes = required_classes or []
        
        self.data_path = self.data_dir / "data"
        self.labels_path = self.data_dir / "labels"
        
        self.logger.info("=" * 50)
        self.logger.info(f"Initializing dataset from {self.data_dir}")
        self.logger.info(f"Class map: {self.class_map}")
        self.logger.info(f"Balance dataset: {self.balance}")
        self.logger.info(f"Required classes: {self.required_classes}")
        self.logger.info("=" * 50)
        
        # Get all valid image/label pairs
        self.file_pairs = self._get_valid_pairs()
        
        if self.balance:
            self.logger.info("Balancing dataset...")
            self.file_pairs = self._balance_dataset()
            
        self.logger.info(f"Dataset initialized with {len(self.file_pairs)} samples")
        if self.balance:
            self.logger.info("Class distribution after balancing:")
            for cls_idx, pairs in self.class_presence.items():
                self.logger.info(f"Class {self.class_map[cls_idx]}: {len(pairs)} samples")
        self.logger.info("=" * 50)
            
    def _get_valid_pairs(self) -> List[Tuple[Path, Path]]:
        """Get all valid image/label pairs and categorize by class presence."""
        valid_pairs = []
        self.class_presence: Dict[int, List[Tuple[Path, Path]]] = {
            cls_idx: [] for cls_idx in self.class_map.keys() if cls_idx != 0  # Exclude background
        }
        
        self.logger.debug("Scanning for valid image/label pairs...")
        total_files = len(list(self.data_path.glob("*.png")))
        valid_count = 0
        
        for img_path in self.data_path.glob("*.png"):
            label_path = self.labels_path / img_path.name
            if label_path.exists():
                # Load label to check class presence
                label = np.array(Image.open(label_path))
                pair = (img_path, label_path)
                valid_pairs.append(pair)
                valid_count += 1
                
                # Record which classes are present in this slice
                for cls_idx in self.class_presence.keys():
                    if np.any(label == cls_idx):
                        self.class_presence[cls_idx].append(pair)
                        
        self.logger.info(f"Found {valid_count}/{total_files} valid image/label pairs")
        return valid_pairs
    
    def _balance_dataset(self) -> List[Tuple[Path, Path]]:
        """Balance dataset based on class presence."""
        if not self.required_classes:
            return self.file_pairs
            
        # Find the minimum number of samples across required classes
        min_samples = min(len(self.class_presence[cls]) 
                         for cls in self.required_classes)
        
        self.logger.info(f"Balancing dataset to {min_samples} samples per class")
        
        # Collect balanced samples
        balanced_pairs = set()
        for cls in self.required_classes:
            pairs = random.sample(self.class_presence[cls], min_samples)
            balanced_pairs.update(pairs)
            self.logger.debug(
                f"Selected {len(pairs)} samples for class {self.class_map[cls]}"
            )
            
        # Add some random negative samples (no required classes)
        negative_pairs = [pair for pair in self.file_pairs 
                         if not any(np.any(np.array(Image.open(pair[1])) == cls) 
                                  for cls in self.required_classes)]
        
        if negative_pairs:
            num_neg = min(len(negative_pairs), min_samples // 2)
            balanced_pairs.update(random.sample(negative_pairs, num_neg))
            self.logger.debug(f"Added {num_neg} negative samples")
            
        return list(balanced_pairs)
    
    def __len__(self):
        return len(self.file_pairs)
        
    def __getitem__(self, idx):
        img_path, label_path = self.file_pairs[idx]
        
        try:
            # Load image and label
            img = Image.open(img_path).convert('L')  # Grayscale
            label = Image.open(label_path)
            
            # Convert to numpy arrays
            img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
            label_np = np.array(label, dtype=np.int64)
            
            # Add channel dimension to image
            img_np = np.expand_dims(img_np, axis=0)
            
            # Apply transforms if provided
            if self.transform:
                img_np, label_np = self.transform(img_np, label_np)
                
            return torch.from_numpy(img_np), torch.from_numpy(label_np)
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            self.logger.error(f"Image path: {img_path}")
            self.logger.error(f"Label path: {label_path}")
            raise

def create_dataloader(
    dataset: Dataset,
    config: DataConfig,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader with the specified configuration."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )

class PairedTransform:
    """Apply consistent transforms to both image and mask."""
    def __init__(self, img_size: Tuple[int, int], augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if self.augment and random.random() > 0.5:
            # Random horizontal flip
            image = np.ascontiguousarray(np.flip(image, axis=2))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))
            
        if self.augment and random.random() > 0.5:
            # Random vertical flip
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=0))
            
        if self.augment and random.random() > 0.5:
            # Random rotation
            angle = random.uniform(-30, 30)
            image = transforms.functional.rotate(
                torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0), 
                angle
            ).squeeze(0).numpy()
            mask = transforms.functional.rotate(
                torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0), 
                angle, 
                interpolation=transforms.InterpolationMode.NEAREST
            ).squeeze(0).numpy()
            
        # Resize to target size
        if image.shape[1:] != self.img_size:
            image = transforms.Resize(self.img_size)(
                torch.from_numpy(np.ascontiguousarray(image))
            ).numpy()
            mask = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )(torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)).squeeze(0).numpy()
            
        # Make sure final outputs are contiguous
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        return image, mask