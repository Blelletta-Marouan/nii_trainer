from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from ..configs.config import DataConfig
from ..utils.logging_utils import setup_logger, log_config

def apply_window(image: np.ndarray, window_width: int, window_level: int) -> np.ndarray:
    """Apply windowing to CT scan data."""
    px_min = window_level - (window_width / 2.0)
    px_max = window_level + (window_width / 2.0)
    clipped = np.clip(image, px_min, px_max)
    return (clipped - px_min) / (px_max - px_min)

def read_nii(filepath: str, rotate: bool = True) -> np.ndarray:
    """Read a NIfTI file and return as numpy array."""
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    if rotate:
        array = np.rot90(array)
    return array

class NiiPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = setup_logger(
            experiment_name="preprocessing",
            save_dir="logs"
        )
        
        # Log configuration using the utility function
        config_dict = {
            "preprocessing": {
                "window_params": self.config.window_params,
                "img_size": self.config.img_size,
                "skip_empty": self.config.skip_empty,
                "slice_step": self.config.slice_step
            }
        }
        log_config(self.logger, config_dict)
        
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess a volume according to configuration."""
        self.logger.info(f"Preprocessing volume of shape {volume.shape}")
        # Apply windowing if CT scan
        if self.config.window_params:
            self.logger.info("Applying intensity windowing...")
            volume = apply_window(
                volume,
                self.config.window_params["window_width"],
                self.config.window_params["window_level"]
            )
        return volume

    def extract_slices(
        self, 
        volume_path: str,
        segmentation_path: str,
        output_dir: Path,
        slice_indices: Optional[List[int]] = None,
        val_split: float = 0.2
    ) -> List[Tuple[Path, Path]]:
        """Extract and save 2D slices from 3D volume."""
        self.logger.info(f"Starting slice extraction from {volume_path}")
        self.logger.info(f"Using segmentation from {segmentation_path}")
        
        # Create train and validation directories with their subdirectories
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        
        # Create data and labels subdirectories for both train and val
        train_data_dir = train_dir / "data"
        train_labels_dir = train_dir / "labels"
        val_data_dir = val_dir / "data"
        val_labels_dir = val_dir / "labels"
        
        for directory in [train_data_dir, train_labels_dir, val_data_dir, val_labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load volume and segmentation
        self.logger.info("Loading NIfTI files...")
        volume = read_nii(volume_path)
        segmentation = read_nii(segmentation_path)
        self.logger.info(f"Loaded volume shape: {volume.shape}")
        self.logger.info(f"Loaded segmentation shape: {segmentation.shape}")
        
        # Preprocess volume
        volume = self.preprocess_volume(volume)
        
        saved_pairs = []
        depth = volume.shape[2]
        
        # Determine which slices to process
        if slice_indices is None:
            slice_indices = range(0, depth, self.config.slice_step)
            self.logger.info(f"Processing every {self.config.slice_step}th slice")
        else:
            self.logger.info(f"Processing specified slice indices: {len(slice_indices)} slices")
        
        skipped_empty = 0
        processed_slices = 0
        
        for idx in slice_indices:
            if idx >= depth:
                continue
                
            slice_img = volume[..., idx]
            slice_seg = segmentation[..., idx]
            
            if self.config.skip_empty and not np.any(slice_seg > 0):
                skipped_empty += 1
                continue
                
            # Convert to uint8 and resize
            img_uint8 = (slice_img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_resized = img_pil.resize(self.config.img_size, Image.BICUBIC)
            
            # Resize segmentation with nearest neighbor
            seg_resized = cv2.resize(
                slice_seg, 
                self.config.img_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Decide train/val split (20% for validation)
            is_validation = np.random.random() < val_split
            
            # Choose appropriate directories based on split
            data_dir = val_data_dir if is_validation else train_data_dir
            labels_dir = val_labels_dir if is_validation else train_labels_dir
            
            # Generate filenames
            base_name = f"slice_{idx:04d}"
            img_path = data_dir / f"{base_name}.png"
            seg_path = labels_dir / f"{base_name}.png"
            
            # Save files
            img_resized.save(img_path)
            cv2.imwrite(str(seg_path), seg_resized.astype(np.uint8))
            
            saved_pairs.append((img_path, seg_path))
            processed_slices += 1
            
            if processed_slices % 50 == 0:
                self.logger.info(f"Processed {processed_slices} slices...")
        
        self.logger.info(f"Finished processing volume:")
        self.logger.info(f"  Total slices processed: {processed_slices}")
        self.logger.info(f"  Empty slices skipped: {skipped_empty}")
        self.logger.info(f"  Total pairs saved: {len(saved_pairs)}")
        
        return saved_pairs