from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from ..configs.config import DataConfig

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
        
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess a volume according to configuration."""
        # Apply windowing if CT scan
        volume = apply_window(
            volume, 
            self.config.window_width,
            self.config.window_level
        )
        return volume

    def extract_slices(
        self, 
        volume_path: str,
        segmentation_path: str,
        output_dir: Path,
        slice_indices: Optional[List[int]] = None
    ) -> List[Tuple[Path, Path]]:
        """
        Extract and save 2D slices from 3D volume.
        Returns list of (image_path, mask_path) tuples.
        """
        # Create output directories
        data_dir = output_dir / "data"
        labels_dir = output_dir / "labels"
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load volume and segmentation
        volume = read_nii(volume_path)
        segmentation = read_nii(segmentation_path)
        
        # Preprocess volume
        volume = self.preprocess_volume(volume)
        
        saved_pairs = []
        depth = volume.shape[2]
        
        # Determine which slices to process
        if slice_indices is None:
            slice_indices = range(0, depth, self.config.slice_step)
            
        for idx in slice_indices:
            if idx >= depth:
                continue
                
            slice_img = volume[..., idx]
            slice_seg = segmentation[..., idx]
            
            if self.config.skip_empty and not np.any(slice_seg > 0):
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
            
            # Generate filenames
            base_name = f"slice_{idx:04d}"
            img_path = data_dir / f"{base_name}.png"
            seg_path = labels_dir / f"{base_name}.png"
            
            # Save files
            img_resized.save(img_path)
            cv2.imwrite(str(seg_path), seg_resized.astype(np.uint8))
            
            saved_pairs.append((img_path, seg_path))
            
        return saved_pairs