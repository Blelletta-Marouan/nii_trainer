from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Callable
import logging
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm
import re

from ..configs.config import DataConfig
from ..utils.logging_utils import setup_logger

def read_nifti_file(filepath: str, rotate: bool = True) -> np.ndarray:
    """
    Read a NIfTI file and return as numpy array.
    
    Args:
        filepath: Path to the NIfTI file
        rotate: Whether to rotate the volume 90 degrees (useful for proper orientation)
        
    Returns:
        Numpy array containing the volume data
    """
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    if rotate:
        array = np.rot90(array)
    return array

def apply_window(image: np.ndarray, window_width: int, window_level: int) -> np.ndarray:
    """
    Apply intensity windowing to CT scan data.
    
    Args:
        image: Input image or volume
        window_width: Width of the intensity window
        window_level: Center level of the intensity window
        
    Returns:
        Windowed image with values normalized to [0, 1]
    """
    px_min = window_level - (window_width / 2.0)
    px_max = window_level + (window_width / 2.0)
    clipped = np.clip(image, px_min, px_max)
    return (clipped - px_min) / (px_max - px_min)

def preprocess_volume(
    volume: np.ndarray, 
    window_params: Optional[Dict[str, int]] = None,
    normalize: bool = True,
    custom_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Preprocess a 3D volume with configurable options.
    
    Args:
        volume: Input 3D volume
        window_params: Optional dict with 'window_width' and 'window_level' for CT windowing
        normalize: Whether to normalize values to [0, 1] if not using windowing
        custom_preprocessor: Optional custom preprocessing function
        
    Returns:
        Preprocessed volume
    """
    # Apply custom preprocessor if provided
    if custom_preprocessor is not None:
        volume = custom_preprocessor(volume)
    
    # Apply windowing for CT scans
    if window_params:
        volume = apply_window(
            volume,
            window_params["window_width"],
            window_params["window_level"]
        )
    # Otherwise, just normalize values if requested
    elif normalize:
        min_val = volume.min()
        max_val = volume.max()
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
            
    return volume

def create_directory_structure(
    output_dir: Union[str, Path],
    include_val: bool = True,
    include_test: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Create a standard directory structure for segmentation datasets.
    
    Args:
        output_dir: Base output directory
        include_val: Whether to create validation directories
        include_test: Whether to create test directories
        
    Returns:
        Dictionary with paths to all created directories
    """
    output_dir = Path(output_dir)
    
    # Define which splits to create
    splits = ["train"]
    if include_val:
        splits.append("val")
    if include_test:
        splits.append("test")
    
    # Create directory structure
    dirs = {}
    for split in splits:
        split_dir = output_dir / split
        data_dir = split_dir / "data"
        labels_dir = split_dir / "labels"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        dirs[split] = {
            "main": split_dir,
            "data": data_dir,
            "labels": labels_dir
        }
    
    return dirs

def extract_and_save_slice(
    slice_img: np.ndarray,
    slice_seg: np.ndarray,
    output_dirs: Dict[str, Dict[str, Path]],
    slice_idx: int,
    target_size: Tuple[int, int],
    split: str,
    volume_id: Optional[str] = None,
    img_interpolation: int = Image.BICUBIC
) -> Tuple[Path, Path]:
    """
    Extract, process and save a single slice with its segmentation.
    
    Args:
        slice_img: Image slice from the volume
        slice_seg: Segmentation slice
        output_dirs: Directory structure from create_directory_structure
        slice_idx: Index of the slice (for naming)
        target_size: Size to resize to (width, height)
        split: Which split this belongs to ('train', 'val', or 'test')
        volume_id: Optional ID of the source volume (for avoiding filename conflicts)
        img_interpolation: Interpolation method for image resizing
        
    Returns:
        Tuple of (image_path, segmentation_path)
    """
    # Generate filename (include volume_id if provided)
    if volume_id is not None:
        base_name = f"vol_{volume_id}_slice_{slice_idx:04d}"
    else:
        base_name = f"slice_{slice_idx:04d}"
    
    # Get output directories
    data_dir = output_dirs[split]["data"]
    labels_dir = output_dirs[split]["labels"]
    
    img_path = data_dir / f"{base_name}.png"
    seg_path = labels_dir / f"{base_name}.png"
    
    # Convert image to uint8 and resize
    img_uint8 = (slice_img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_resized = img_pil.resize(target_size, img_interpolation)
    
    # Resize segmentation with nearest neighbor
    seg_resized = cv2.resize(
        slice_seg, 
        target_size, 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Save files
    img_resized.save(img_path)
    cv2.imwrite(str(seg_path), seg_resized.astype(np.uint8))
    
    return img_path, seg_path

def process_volume_and_segmentation(
    volume_path: str,
    segmentation_path: str,
    output_dir: Union[str, Path],
    img_size: Tuple[int, int] = (256, 256),
    slice_indices: Optional[List[int]] = None,
    slice_step: int = 1,
    skip_empty: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_params: Optional[Dict[str, int]] = None,
    custom_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_seg_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    force_overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[Path, Path]]:
    """
    Comprehensive function to process a volume and its segmentation.
    
    Args:
        volume_path: Path to the NIfTI volume file
        segmentation_path: Path to the NIfTI segmentation file
        output_dir: Directory to save processed data
        img_size: Target image size (width, height)
        slice_indices: Specific indices to extract (if None, use slice_step)
        slice_step: How many slices to skip between extractions
        skip_empty: Whether to skip slices without segmentation labels
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        window_params: CT windowing parameters (dict with 'window_width' and 'window_level')
        custom_preprocessor: Optional custom preprocessing function for volume
        custom_seg_processor: Optional custom preprocessing function for segmentation
        force_overwrite: Whether to overwrite existing files
        logger: Optional logger instance
        
    Returns:
        List of tuples containing (image_path, segmentation_path) pairs
    """
    # Setup logger if not provided
    if logger is None:
        logger = setup_logger(
            experiment_name="volume_processing",
            save_dir="logs"
        )
    
    logger.info(f"Processing volume: {volume_path}")
    logger.info(f"With segmentation: {segmentation_path}")
    
    # Create directory structure
    output_dir = Path(output_dir)
    output_dirs = create_directory_structure(
        output_dir, 
        include_val=(val_ratio > 0),
        include_test=(test_ratio > 0)
    )
    
    # Load volume and segmentation
    logger.info("Loading NIfTI files...")
    volume = read_nifti_file(volume_path)
    segmentation = read_nifti_file(segmentation_path)
    
    if volume.shape != segmentation.shape:
        logger.warning(f"Volume shape {volume.shape} doesn't match segmentation shape {segmentation.shape}")
    
    logger.info(f"Volume shape: {volume.shape}")
    
    # Preprocess volume
    volume = preprocess_volume(
        volume, 
        window_params=window_params,
        custom_preprocessor=custom_preprocessor
    )
    
    # Apply custom segmentation processing if provided
    if custom_seg_processor is not None:
        segmentation = custom_seg_processor(segmentation)
    
    # Determine which slices to process
    depth = volume.shape[2]
    if slice_indices is None:
        slice_indices = range(0, depth, slice_step)
        logger.info(f"Processing every {slice_step}th slice")
    else:
        logger.info(f"Processing {len(slice_indices)} specified slice indices")
    
    # Extract volume ID from filename
    volume_id = None
    volume_match = re.match(r'.*volume-(\d+)\.nii', os.path.basename(volume_path))
    if volume_match:
        volume_id = volume_match.group(1)
    else:
        # Try to extract any numeric ID from the filename
        volume_match = re.search(r'(\d+)', os.path.basename(volume_path))
        if volume_match:
            volume_id = volume_match.group(1)
        else:
            # Use a hash of the filepath as last resort
            volume_id = str(abs(hash(volume_path)) % 10000)
            
    logger.info(f"Using volume ID: {volume_id}")
    
    # Initialize statistics counters
    saved_pairs = []
    skipped_empty = 0
    skipped_existing = 0
    processed_slices = 0
    
    # Initialize counters for each split
    train_count = val_count = test_count = 0
    
    # Normalize split ratios if they don't sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        logger.info(f"Normalized split ratios to: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    
    # Process slices with progress bar
    for idx in tqdm(list(slice_indices), desc="Processing slices"):
        if idx >= depth:
            continue
        
        # Extract slice
        slice_img = volume[..., idx]
        slice_seg = segmentation[..., idx]
        
        # Skip empty slices if requested
        if skip_empty and not np.any(slice_seg > 0):
            skipped_empty += 1
            continue
        
        # Generate base name for the slice (with volume ID)
        if volume_id is not None:
            base_name = f"vol_{volume_id}_slice_{idx:04d}"
        else:
            base_name = f"slice_{idx:04d}"
        
        # Check if files already exist
        file_exists = False
        for split_name in output_dirs.keys():
            img_path = output_dirs[split_name]["data"] / f"{base_name}.png"
            seg_path = output_dirs[split_name]["labels"] / f"{base_name}.png"
            
            if img_path.exists() and seg_path.exists() and not force_overwrite:
                file_exists = True
                saved_pairs.append((img_path, seg_path))
                skipped_existing += 1
                break
        
        if file_exists:
            continue
        
        # Determine split based on ratios
        random_val = random.random()
        if random_val < train_ratio:
            split = "train"
            train_count += 1
        elif random_val < train_ratio + val_ratio:
            split = "val"
            val_count += 1
        else:
            split = "test"
            test_count += 1
        
        # Extract and save the slice
        img_path, seg_path = extract_and_save_slice(
            slice_img, 
            slice_seg,
            output_dirs,
            idx,
            img_size,
            split,
            volume_id  # Pass the volume ID to the function
        )
        
        saved_pairs.append((img_path, seg_path))
        processed_slices += 1
    
    # Log final statistics
    logger.info(f"Finished processing volume:")
    logger.info(f"  Total slices processed: {processed_slices}")
    logger.info(f"  Empty slices skipped: {skipped_empty}")
    logger.info(f"  Existing slices skipped: {skipped_existing}")
    logger.info(f"  Split distribution:")
    logger.info(f"    Train: {train_count} ({train_count/max(processed_slices,1)*100:.1f}%)")
    logger.info(f"    Val: {val_count} ({val_count/max(processed_slices,1)*100:.1f}%)")
    logger.info(f"    Test: {test_count} ({test_count/max(processed_slices,1)*100:.1f}%)")
    logger.info(f"  Total pairs saved: {len(saved_pairs)}")
    
    return saved_pairs