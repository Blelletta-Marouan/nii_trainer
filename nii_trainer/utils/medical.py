"""
Medical imaging specific utilities for NII-Trainer.
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, Any, Union
from ..core.exceptions import NIITrainerError


def get_image_info(image: sitk.Image) -> Dict[str, Any]:
    """Get comprehensive information about a medical image."""
    return {
        'size': image.GetSize(),
        'origin': image.GetOrigin(),
        'spacing': image.GetSpacing(),
        'direction': image.GetDirection(),
        'pixel_type': image.GetPixelIDTypeAsString(),
        'number_of_components': image.GetNumberOfComponentsPerPixel(),
        'dimension': image.GetDimension(),
        'physical_size': [s * sp for s, sp in zip(image.GetSize(), image.GetSpacing())]
    }


def resample_image(
    image: sitk.Image,
    new_spacing: Tuple[float, ...],
    new_size: Optional[Tuple[int, ...]] = None,
    interpolator: int = sitk.sitkLinear,
    fill_value: float = 0.0
) -> sitk.Image:
    """Resample image to new spacing and optionally new size."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    if new_size is None:
        # Calculate new size based on spacing change
        new_size = tuple([
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(len(original_size))
        ])
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(fill_value)
    
    return resampler.Execute(image)


def crop_to_nonzero(
    image: sitk.Image,
    label: Optional[sitk.Image] = None,
    margin: int = 0
) -> Tuple[sitk.Image, Optional[sitk.Image], Tuple[slice, ...]]:
    """Crop image to non-zero region with optional margin."""
    # Convert to numpy for processing
    image_array = sitk.GetArrayFromImage(image)
    
    # Find bounding box of non-zero values
    nonzero_indices = np.nonzero(image_array)
    if len(nonzero_indices[0]) == 0:
        raise NIITrainerError("Image contains only zero values")
    
    # Get min/max coordinates
    min_coords = [np.min(indices) for indices in nonzero_indices]
    max_coords = [np.max(indices) for indices in nonzero_indices]
    
    # Add margin
    cropping_slices = []
    for i, (min_c, max_c) in enumerate(zip(min_coords, max_coords)):
        start = max(0, min_c - margin)
        end = min(image_array.shape[i], max_c + margin + 1)
        cropping_slices.append(slice(start, end))
    
    cropping_slices = tuple(cropping_slices)
    
    # Crop image
    cropped_array = image_array[cropping_slices]
    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.CopyInformation(image)
    
    # Crop label if provided
    cropped_label = None
    if label is not None:
        label_array = sitk.GetArrayFromImage(label)
        cropped_label_array = label_array[cropping_slices]
        cropped_label = sitk.GetImageFromArray(cropped_label_array)
        cropped_label.CopyInformation(label)
    
    return cropped_image, cropped_label, cropping_slices


def pad_image(
    image: sitk.Image,
    target_size: Tuple[int, ...],
    fill_value: float = 0.0,
    center: bool = True
) -> sitk.Image:
    """Pad image to target size."""
    current_size = image.GetSize()
    
    if len(current_size) != len(target_size):
        raise ValueError("Image and target size dimensions must match")
    
    # Calculate padding
    padding = []
    for i, (current, target) in enumerate(zip(current_size, target_size)):
        if target < current:
            raise ValueError(f"Target size {target} smaller than current size {current} at dimension {i}")
        
        total_pad = target - current
        if center:
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:
            pad_before = 0
            pad_after = total_pad
        
        padding.extend([pad_before, pad_after])
    
    # Apply padding
    padder = sitk.ConstantPadImageFilter()
    padder.SetPadLowerBound([padding[i*2] for i in range(len(target_size))])
    padder.SetPadUpperBound([padding[i*2+1] for i in range(len(target_size))])
    padder.SetConstant(fill_value)
    
    return padder.Execute(image)


def compute_bounding_box(
    mask: sitk.Image,
    margin: int = 0
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute bounding box of non-zero region in mask."""
    mask_array = sitk.GetArrayFromImage(mask)
    nonzero_indices = np.nonzero(mask_array)
    
    if len(nonzero_indices[0]) == 0:
        return tuple([0] * mask.GetDimension()), tuple(mask.GetSize())
    
    min_coords = tuple([max(0, np.min(indices) - margin) for indices in nonzero_indices])
    max_coords = tuple([min(mask_array.shape[i], np.max(nonzero_indices[i]) + margin + 1) 
                       for i in range(len(nonzero_indices))])
    
    return min_coords, max_coords


def apply_window_level(
    image_array: np.ndarray,
    window_center: float,
    window_width: float,
    output_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Apply window/level transformation to image."""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    # Clip values
    windowed = np.clip(image_array, min_val, max_val)
    
    # Normalize to output range
    if max_val > min_val:
        windowed = (windowed - min_val) / (max_val - min_val)
        windowed = windowed * (output_range[1] - output_range[0]) + output_range[0]
    
    return windowed


def normalize_hu_values(
    image_array: np.ndarray,
    hu_min: float = -1000,
    hu_max: float = 1000,
    output_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Normalize HU values to specified range."""
    clipped = np.clip(image_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]
    return normalized


def convert_coordinates(
    coordinates: Tuple[float, ...],
    from_image: sitk.Image,
    to_image: sitk.Image
) -> Tuple[float, ...]:
    """Convert coordinates from one image space to another."""
    # Convert to physical coordinates
    physical_coords = from_image.TransformIndexToPhysicalPoint(coordinates)
    
    # Convert to target image coordinates
    target_coords = to_image.TransformPhysicalPointToIndex(physical_coords)
    
    return target_coords


def get_image_orientation(image: sitk.Image) -> str:
    """Get image orientation string (e.g., 'RAS', 'LPS')."""
    direction = np.array(image.GetDirection()).reshape((image.GetDimension(), -1))
    orientation = ""
    
    for i in range(image.GetDimension()):
        axis = direction[i]
        max_idx = np.argmax(np.abs(axis))
        
        if max_idx == 0:  # X axis
            orientation += "R" if axis[0] > 0 else "L"
        elif max_idx == 1:  # Y axis
            orientation += "A" if axis[1] > 0 else "P"
        elif max_idx == 2:  # Z axis
            orientation += "S" if axis[2] > 0 else "I"
    
    return orientation


def compute_volume_stats(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None
) -> Dict[str, float]:
    """Compute statistical measures of image volume."""
    image_array = sitk.GetArrayFromImage(image)
    
    if mask is not None:
        mask_array = sitk.GetArrayFromImage(mask)
        image_array = image_array[mask_array > 0]
    
    stats = {
        'mean': float(np.mean(image_array)),
        'std': float(np.std(image_array)),
        'min': float(np.min(image_array)),
        'max': float(np.max(image_array)),
        'median': float(np.median(image_array)),
        'percentile_1': float(np.percentile(image_array, 1)),
        'percentile_99': float(np.percentile(image_array, 99)),
        'volume_ml': float(np.prod(image.GetSpacing()) * len(image_array) / 1000)
    }
    
    return stats