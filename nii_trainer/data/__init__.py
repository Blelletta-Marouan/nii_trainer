"""
Data processing module for NII-Trainer.
"""

from .readers import (
    BaseReader,
    NIfTIReader,
    DICOMReader,
    NRRDReader,
    GenericITKReader,
    ReaderFactory,
    reader_factory,
    read_medical_image
)

from .processors import (
    VolumeProcessor,
    SliceExtractor,
    MorphologicalProcessor,
    WINDOWING_PRESETS,
    get_windowing_preset
)

from .transforms import (
    BaseTransform,
    RandomRotation,
    RandomTranslation,
    RandomScale,
    RandomFlip,
    GaussianNoise,
    IntensityShift,
    IntensityScale,
    GammaCorrection,
    ElasticDeformation,
    Compose,
    OneOf,
    get_training_transforms,
    get_validation_transforms,
    TRANSFORM_PRESETS,
    get_transform_preset
)

__all__ = [
    # Readers
    "BaseReader", "NIfTIReader", "DICOMReader", "NRRDReader", 
    "GenericITKReader", "ReaderFactory", "reader_factory", "read_medical_image",
    
    # Processors
    "VolumeProcessor", "SliceExtractor", "MorphologicalProcessor", 
    "WINDOWING_PRESETS", "get_windowing_preset",
    
    # Transforms
    "BaseTransform", "RandomRotation", "RandomTranslation", "RandomScale",
    "RandomFlip", "GaussianNoise", "IntensityShift", "IntensityScale",
    "GammaCorrection", "ElasticDeformation", "Compose", "OneOf",
    "get_training_transforms", "get_validation_transforms", 
    "TRANSFORM_PRESETS", "get_transform_preset"
]