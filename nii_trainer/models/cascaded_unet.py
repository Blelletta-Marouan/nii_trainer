import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .base import EncoderFactory, DecoderBlock
from ..configs.config import CascadeConfig, StageConfig

class CascadeStage(nn.Module):
    """
    A single stage in the cascade, consisting of a UNet with:
    - Configurable encoder backbone
    - Flexible number of encoder/decoder layers
    - Skip connections (optional)
    - Support for binary segmentation in first stage
    """
    def __init__(
        self,
        in_channels: int,
        initial_features: int,
        stage_config: StageConfig,
        pretrained: bool = True
    ):
        super().__init__()
        self.config = stage_config
        
        # Create encoder
        self.encoder = self._create_encoder(
            in_channels,
            initial_features,
            stage_config,
            pretrained
        )
        
        # Get encoder output channels
        encoder_channels = self.encoder.out_channels
        
        # Create decoder stages
        self.decoder = self._create_decoder(
            encoder_channels,
            stage_config.decoder_layers,
            stage_config.dropout_rate
        )
        
        # For binary segmentation (first stage), we use sigmoid
        # For multi-class stages, we use softmax if needed
        self.is_binary = stage_config.is_binary
        
    def _create_encoder(
        self,
        in_channels: int,
        initial_features: int,
        config: StageConfig,
        pretrained: bool
    ) -> nn.Module:
        """Create encoder with proper configuration."""
        return EncoderFactory.create_encoder({
            "encoder_type": config.encoder_type,
            "in_channels": in_channels,
            "initial_features": initial_features,
            "num_layers": config.encoder_layers,
            "pretrained": pretrained
        })
        
    def _create_decoder(
        self,
        encoder_channels: List[int],
        num_layers: int,
        dropout_rate: float
    ) -> nn.ModuleList:
        """Create decoder with specified number of layers."""
        decoder = nn.ModuleList()
        reversed_channels = encoder_channels[::-1]
        
        for i in range(num_layers - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1] if self.config.skip_connections else 0
            out_ch = reversed_channels[i + 1]
            
            decoder.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout_rate=dropout_rate
                )
            )
            
        # For binary segmentation, output 1 channel
        # For multi-class, output number of classes in stage's input_classes
        out_channels = 1 if self.is_binary else len(self.config.input_classes)
        decoder.append(nn.Conv2d(reversed_channels[-1], out_channels, kernel_size=1))
        return decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single stage."""
        # Encode
        enc_features = self.encoder(x)
        
        # Decode
        dec_features = enc_features[::-1]  # Reverse for decoder
        out = dec_features[0]
        
        for i, dec_block in enumerate(self.decoder[:-1]):
            skip = dec_features[i + 1] if self.config.skip_connections else None
            out = dec_block(out, skip)
            
        # Final convolution
        out = self.decoder[-1](out)
        
        # For binary stages, we don't apply activation (handled by loss)
        # For multi-class stages, we could apply softmax if needed
        return out

class FlexibleCascadedUNet(nn.Module):
    """
    A flexible cascaded UNet that supports:
    - Binary segmentation in first stage
    - Arbitrary number of cascade stages
    - Different encoder backbones per stage
    - Configurable number of layers per stage
    - Flexible class handling per stage
    """
    def __init__(self, config: CascadeConfig):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        
        # Create each stage
        for stage_idx, stage_config in enumerate(config.stages):
            # Calculate input channels for this stage
            if stage_idx == 0:
                stage_in_channels = config.in_channels
            else:
                # Input channels include original input plus outputs from all previous stages
                stage_in_channels = config.in_channels + stage_idx
                
            # Create stage
            stage = CascadeStage(
                in_channels=stage_in_channels,
                initial_features=int(config.initial_features * (config.feature_growth ** stage_idx)),
                stage_config=stage_config,
                pretrained=config.pretrained
            )
            self.stages.append(stage)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning outputs from all stages.
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            List of outputs from each stage [B, 1, H, W] or [B, num_classes, H, W]
        """
        stage_outputs = []
        stage_input = x
        
        for stage in self.stages:
            # Current stage prediction
            stage_out = stage(stage_input)
            stage_outputs.append(stage_out)
            
            # For first stage (binary), convert output to binary mask
            if stage.is_binary:
                binary_mask = torch.sigmoid(stage_out) > stage.config.threshold
                stage_input = torch.cat([x, binary_mask.float()], dim=1)
            else:
                # For multi-class stages, concatenate probabilities
                stage_input = torch.cat([x] + stage_outputs, dim=1)
                
        return stage_outputs
    
    @staticmethod
    def get_predictions(
        outputs: List[torch.Tensor],
        thresholds: List[float]
    ) -> torch.Tensor:
        """
        Convert model outputs to final predictions using thresholds.
        First stage uses binary threshold, subsequent stages use class-specific thresholds.
        Returns tensor of shape [B, H, W] with class indices.
        """
        batch_size = outputs[0].shape[0]
        height = outputs[0].shape[2]
        width = outputs[0].shape[3]
        
        # Start with zeros (background)
        final_mask = torch.zeros(
            (batch_size, height, width),
            device=outputs[0].device,
            dtype=torch.long
        )
        
        # First stage: binary foreground/background
        binary_mask = (torch.sigmoid(outputs[0]) > thresholds[0]).squeeze(1)
        final_mask[binary_mask] = 1
        
        # Subsequent stages: class-specific segmentation
        for i, (out, thresh) in enumerate(list(zip(outputs[1:], thresholds[1:]))):
            prob = torch.sigmoid(out)
            mask = (prob >= thresh).squeeze(1)
            # Only apply within the foreground region from first stage
            final_mask[mask & binary_mask] = i + 2
            
        return final_mask