from pathlib import Path
import logging
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import glob
import re
import numpy as np
from tqdm import tqdm
import os
import time

from .nifti_processor import (
    read_nifti_file, 
    apply_window,
    preprocess_volume, 
    process_volume_and_segmentation,
    create_directory_structure
)
from ..configs.config import DataConfig
from ..utils.logging_utils import setup_logger


class BatchProcessor:
    """
    A class for batch processing of multiple NIfTI volumes and segmentations.
    
    This class handles the preprocessing of multiple volume-segmentation pairs
    according to configured naming patterns and directory structures.
    """
    
    def __init__(
        self,
        config: Optional[DataConfig] = None,
        img_size: Optional[Tuple[int, int]] = None,
        window_params: Optional[Dict[str, int]] = None,
        skip_empty: bool = False,
        slice_step: int = 1,
        train_val_test_split: Optional[Tuple[float, float, float]] = None,
        logger: Optional[logging.Logger] = None,
        segmentation_pattern: str = "segmentation-{}.nii",
        volume_pattern: str = "*-{}.nii"
    ):
        """
        Initialize the BatchProcessor with highly customizable options.
        
        Args:
            config: Configuration object with preprocessing parameters
            img_size: Target image size (width, height) - overrides config
            window_params: CT windowing parameters (dict with 'window_width' and 'window_level')
            skip_empty: Whether to skip slices without annotations
            slice_step: How many slices to skip between extractions
            train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio)
            logger: Optional logger to use (creates one if not provided)
            segmentation_pattern: Pattern for matching segmentation files
                                 (use {} as a placeholder for the volume ID)
            volume_pattern: Pattern for matching volume files
                           (use {} as a placeholder for the volume ID)
        """
        self.config = config
        
        # Allow direct parameter specification instead of requiring config
        self.img_size = img_size or (config.img_size if config else (256, 256))
        self.window_params = window_params or (config.window_params if config else None)
        self.skip_empty = skip_empty if skip_empty is not None else (config.skip_empty if config else False)
        self.slice_step = slice_step or (config.slice_step if config else 1)
        self.train_val_test_split = train_val_test_split or (config.train_val_test_split if config else (0.7, 0.15, 0.15))
        
        # Initialize logger
        self.logger = logger or setup_logger(
            experiment_name="batch_processing",
            save_dir="logs"
        )
        
        # Set naming patterns
        self.segmentation_pattern = segmentation_pattern
        self.volume_pattern = volume_pattern
        
        # Log configuration
        self.logger.info(f"BatchProcessor initialized with:")
        self.logger.info(f"  Image size: {self.img_size}")
        self.logger.info(f"  Window params: {self.window_params}")
        self.logger.info(f"  Skip empty: {self.skip_empty}")
        self.logger.info(f"  Slice step: {self.slice_step}")
        self.logger.info(f"  Train/Val/Test split: {self.train_val_test_split}")
        self.logger.info(f"  Segmentation pattern: {self.segmentation_pattern}")
        self.logger.info(f"  Volume pattern: {self.volume_pattern}")
        
    def _extract_id(self, filename: str, pattern: str) -> Optional[str]:
        """
        Extract the ID from a filename using a pattern.
        
        Args:
            filename: The filename to extract from
            pattern: Pattern with {} placeholder for the ID
            
        Returns:
            Extracted ID or None if no match
        """
        # Special case for "volume-X.nii" pattern - extract the number directly
        volume_match = re.match(r'volume-(\d+)\.nii', filename)
        if volume_match:
            return volume_match.group(1)
            
        # For other patterns, use the generic approach
        regex = pattern.replace(".", "\\.").replace("*", ".*").replace("{}", "(.*)")
        match = re.match(regex, filename)
        return match.group(1) if match else None
    
    def _find_matching_segmentation(
        self, 
        volume_path: Path, 
        segmentation_dir: Path
    ) -> Optional[Path]:
        """
        Find matching segmentation file for a volume.
        
        Args:
            volume_path: Path to the volume file
            segmentation_dir: Directory containing segmentation files
            
        Returns:
            Path to matching segmentation file or None if not found
        """
        # Extract ID from volume filename
        volume_filename = volume_path.name
        volume_id = self._extract_id(volume_filename, self.volume_pattern.replace("*", ""))

        if volume_id is None:
            self.logger.warning(f"Could not extract ID from volume: {volume_filename}")
            return None
            
        # Construct segmentation filename
        seg_filename = self.segmentation_pattern.format(volume_id)
        seg_path = segmentation_dir / seg_filename
        
        if not seg_path.exists():
            self.logger.warning(f"No matching segmentation found for volume ID {volume_id}")
            return None
            
        return seg_path
    
    def process_volume(
        self,
        volume_path: Union[str, Path],
        segmentation_path: Union[str, Path],
        output_dir: Union[str, Path],
        include_val: bool = True,
        include_test: bool = True,
        custom_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        custom_seg_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        force_overwrite: bool = False,
        **kwargs
    ) -> List[Tuple[Path, Path]]:
        """
        Process a single volume and its segmentation.
        
        Args:
            volume_path: Path to volume file
            segmentation_path: Path to segmentation file
            output_dir: Directory to save processed data
            include_val: Whether to create validation split
            include_test: Whether to create test split
            custom_preprocessor: Optional custom function for volume preprocessing
            custom_seg_processor: Optional custom function for segmentation preprocessing
            force_overwrite: Whether to overwrite existing files
            **kwargs: Additional arguments to pass to process_volume_and_segmentation
            
        Returns:
            List of tuples containing (image_path, segmentation_path) pairs
        """
        # Extract the split ratios
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        
        # Adjust ratios if not including validation or test
        if not include_val:
            train_ratio += val_ratio
            val_ratio = 0.0
        if not include_test:
            train_ratio += test_ratio
            test_ratio = 0.0
            
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # Use our new modular process_volume_and_segmentation function
        return process_volume_and_segmentation(
            volume_path=str(volume_path),
            segmentation_path=str(segmentation_path),
            output_dir=output_dir,
            img_size=self.img_size,
            slice_step=self.slice_step,
            skip_empty=self.skip_empty,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            window_params=self.window_params,
            custom_preprocessor=custom_preprocessor,
            custom_seg_processor=custom_seg_processor,
            force_overwrite=force_overwrite,
            logger=self.logger,
            **kwargs
        )
        
    def process_batch(
        self,
        volume_dir: Union[str, Path],
        segmentation_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.nii",
        max_volumes: Optional[int] = None,
        skip_existing: bool = True,
        include_val: bool = True,
        include_test: bool = True,
        custom_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        custom_seg_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        force_overwrite: bool = False
    ) -> List[Tuple[Path, Path]]:
        """
        Process a batch of volumes and their segmentations.
        
        Args:
            volume_dir: Directory containing volume files
            segmentation_dir: Directory containing segmentation files
            output_dir: Directory to save processed data
            file_pattern: Pattern to match volume files
            max_volumes: Maximum number of volumes to process (None for all)
            skip_existing: Skip volumes that have already been processed
            include_val: Whether to create validation split
            include_test: Whether to create test split
            custom_preprocessor: Optional custom function for volume preprocessing
            custom_seg_processor: Optional custom function for segmentation preprocessing
            force_overwrite: Whether to overwrite existing files
            
        Returns:
            List of tuples containing (image_path, segmentation_path) pairs
        """
        # Convert to Path objects
        volume_dir = Path(volume_dir)
        segmentation_dir = Path(segmentation_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all volume files
        volume_files = sorted(list(volume_dir.glob(file_pattern)))
        self.logger.info(f"Found {len(volume_files)} volumes matching pattern '{file_pattern}'")
        
        # Limit number of volumes if specified
        if max_volumes is not None:
            volume_files = volume_files[:max_volumes]
            self.logger.info(f"Limited processing to {max_volumes} volumes")
        
        # Process volumes
        processed_pairs = []
        
        # Track timing for performance analysis
        start_time = time.time()
        
        # Use tqdm for progress tracking of volumes
        for volume_index, volume_path in enumerate(tqdm(volume_files, desc="Processing volumes", unit="volume")):
            self.logger.info(f"Processing volume {volume_index+1}/{len(volume_files)}: {volume_path.name}")
            
            # Find matching segmentation file
            seg_path = self._find_matching_segmentation(volume_path, segmentation_dir)
            
            if seg_path is None:
                continue
                
            self.logger.info(f"Found matching segmentation: {seg_path.name}")
            
            # Process this volume-segmentation pair
            volume_start_time = time.time()
            pairs = self.process_volume(
                volume_path=volume_path,
                segmentation_path=seg_path,
                output_dir=output_dir,
                include_val=include_val,
                include_test=include_test,
                custom_preprocessor=custom_preprocessor,
                custom_seg_processor=custom_seg_processor,
                force_overwrite=force_overwrite
            )
            
            # Add to processed pairs
            processed_pairs.extend(pairs)
            volume_time = time.time() - volume_start_time
            self.logger.info(f"Extracted {len(pairs)} valid slices from volume in {volume_time:.2f} seconds")
        
        # Add some visual separation in the console
        total_time = time.time() - start_time
        print("\n" + "="*80)
        self.logger.info(f"Batch processing complete in {total_time:.2f} seconds")
        self.logger.info(f"Processed {len(volume_files)} volumes")
        self.logger.info(f"Total processed image pairs: {len(processed_pairs)}")
        if len(volume_files) > 0:
            self.logger.info(f"Average time per volume: {total_time/len(volume_files):.2f} seconds")
        print("="*80 + "\n")
        
        return processed_pairs
        
    def process_with_naming_convention(
        self,
        base_dir: Union[str, Path],
        segmentation_dir: Union[str, Path],
        output_dir: Union[str, Path],
        naming_convention: Dict[str, str] = None,
        **kwargs
    ) -> List[Tuple[Path, Path]]:
        """
        Process volumes using custom naming conventions.
        
        Args:
            base_dir: Directory containing volume files
            segmentation_dir: Directory containing segmentation files
            output_dir: Directory to save processed data
            naming_convention: Custom naming convention mapping
                             (e.g., {"volume": "vol-{}.nii", "segmentation": "seg-{}.nii"})
            **kwargs: Additional arguments to pass to process_batch
            
        Returns:
            List of tuples containing (image_path, segmentation_path) pairs
        """
        if naming_convention:
            if "volume" in naming_convention:
                self.volume_pattern = naming_convention["volume"]
            if "segmentation" in naming_convention:
                self.segmentation_pattern = naming_convention["segmentation"]
                
        self.logger.info(f"Using volume pattern: {self.volume_pattern}")
        self.logger.info(f"Using segmentation pattern: {self.segmentation_pattern}")
        
        return self.process_batch(
            volume_dir=base_dir,
            segmentation_dir=segmentation_dir,
            output_dir=output_dir,
            **kwargs
        )