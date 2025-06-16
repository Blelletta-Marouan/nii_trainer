"""
Models module for NII-Trainer.
"""

from .base import (
    BaseModel,
    BaseEncoder,
    BaseDecoder,
    BaseCascadeModel,
    StageModule,
    AttentionFusion,
    calculate_receptive_field,
    get_model_summary
)

from .encoders import (
    ResNet18Encoder,
    ResNet50Encoder,
    EfficientNetB0Encoder,
    MedicalCNNEncoder,
    AttentionEncoder,
    ChannelAttention,
    SpatialAttention,
    create_encoder
)

from .decoders import (
    UNetDecoder,
    FPNDecoder,
    DeepLabV3Decoder,
    LinkNetDecoder,
    AttentionGate,
    ASPP,
    LinkNetDecoderBlock,
    create_decoder
)

from .cascaded import (
    CascadedSegmentationModel,
    SimpleUNet,
    ProgressiveCascadeModel,
    ProgressiveStage,
    create_model,
    load_pretrained_model
)

__all__ = [
    # Base classes
    "BaseModel", "BaseEncoder", "BaseDecoder", "BaseCascadeModel",
    "StageModule", "AttentionFusion", "calculate_receptive_field", "get_model_summary",
    
    # Encoders
    "ResNet18Encoder", "ResNet50Encoder", "EfficientNetB0Encoder",
    "MedicalCNNEncoder", "AttentionEncoder", "ChannelAttention", 
    "SpatialAttention", "create_encoder",
    
    # Decoders
    "UNetDecoder", "FPNDecoder", "DeepLabV3Decoder", "LinkNetDecoder",
    "AttentionGate", "ASPP", "LinkNetDecoderBlock", "create_decoder",
    
    # Complete models
    "CascadedSegmentationModel", "SimpleUNet", "ProgressiveCascadeModel",
    "ProgressiveStage", "create_model", "load_pretrained_model"
]