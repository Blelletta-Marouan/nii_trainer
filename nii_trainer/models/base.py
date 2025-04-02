from typing import Dict, List, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ..configs.config import ModelConfig

class EncoderFactory:
    @staticmethod
    def create_encoder(config: ModelConfig) -> nn.Module:
        """Create encoder based on configuration."""
        if config.encoder_type == "mobilenet_v2":
            return MobileNetV2Encoder(config)
        elif config.encoder_type == "resnet18":
            return ResNetEncoder(
                model_fn=models.resnet18,
                in_channels=config.in_channels,
                initial_features=config.initial_features,
                num_layers=config.num_layers,
                pretrained=config.pretrained
            )
        elif config.encoder_type == "efficientnet":
            return EfficientNetEncoder(config)
        else:
            raise ValueError(f"Unsupported encoder type: {config.encoder_type}")

class EncoderBase(nn.Module):
    """Base class for all encoders."""
    def __init__(self):
        super().__init__()
        self.stages: nn.ModuleList
        self.out_channels: List[int]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class MobileNetV2Encoder(EncoderBase):
    """MobileNetV2-based encoder."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=config.pretrained)
        
        # Modify first conv for arbitrary input channels
        first_conv = nn.Conv2d(config.in_channels, 32, kernel_size=3, 
                             stride=2, padding=1, bias=False)
        if config.pretrained:
            # Initialize with pretrained weights
            if config.in_channels == 1:
                # For single channel input, average the RGB weights
                with torch.no_grad():
                    first_conv.weight.data = backbone.features[0][0].weight.data.mean(dim=1, keepdim=True)
            elif config.in_channels == 3:
                # Standard 3-channel case, directly copy weights
                with torch.no_grad():
                    first_conv.weight[:, :3, ...] = backbone.features[0][0].weight
            elif config.in_channels > 3:
                # Multi-channel case: copy first 3 channels and initialize the rest
                with torch.no_grad():
                    first_conv.weight[:, :3, ...] = backbone.features[0][0].weight
                    mean_weights = torch.mean(backbone.features[0][0].weight, dim=1)
                    first_conv.weight[:, 3:, ...] = mean_weights.unsqueeze(1)
                    
        backbone.features[0][0] = first_conv
        
        # Split backbone into stages
        self.stages = nn.ModuleList([
            backbone.features[0],  # stage 1: 32 channels
            nn.Sequential(*backbone.features[1:3]),  # stage 2: 24 channels
            nn.Sequential(*backbone.features[3:6]),  # stage 3: 32 channels
            nn.Sequential(*backbone.features[6:13]),  # stage 4: 96 channels
            nn.Sequential(*backbone.features[13:])  # stage 5: 320 channels
        ])
        
        self.out_channels = [32, 24, 32, 96, 320]

class ResNetEncoder(EncoderBase):
    """ResNet-based encoder."""
    def __init__(
        self,
        model_fn: Type[nn.Module],
        in_channels: int,
        initial_features: int,
        num_layers: int,
        pretrained: bool = True
    ):
        super().__init__()
        # Initialize with pretrained weights if requested
        backbone = model_fn(pretrained=pretrained)
        
        # Modify first convolution layer to handle different input channels
        first_conv = nn.Conv2d(
            in_channels,
            64,  # ResNet always uses 64 initial features
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # If using pretrained weights, adapt the first conv layer
        if pretrained:
            # For single channel input, average the RGB weights
            if in_channels == 1:
                first_conv.weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                first_conv.weight.data[:, :3, ...] = backbone.conv1.weight.data
                first_conv.weight.data[:, 3:, ...] = 0.0
        
        backbone.conv1 = first_conv
        
        # Create stages
        self.stages = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ][:num_layers])  # Only keep the requested number of layers
        
        # Set output channels based on ResNet variant
        if "18" in str(model_fn) or "34" in str(model_fn):
            self.out_channels = [64, 64, 128, 256, 512][:num_layers]
        else:  # ResNet50, 101, 152
            self.out_channels = [64, 256, 512, 1024, 2048][:num_layers]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class EfficientNetEncoder(EncoderBase):
    """EfficientNet-based encoder."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=config.pretrained)
        
        # Modify first conv for arbitrary input channels
        if config.in_channels != 3:
            first_conv = nn.Conv2d(config.in_channels, 32, kernel_size=3,
                                 stride=2, padding=1, bias=False)
            if config.pretrained:
                with torch.no_grad():
                    # For single channel, use mean of RGB weights
                    if config.in_channels == 1:
                        first_conv.weight.data = backbone.features[0][0].weight.data.mean(dim=1, keepdim=True)
                    else:
                        # For more channels, copy first 3 and initialize rest with mean
                        first_conv.weight[:, :min(3, config.in_channels), ...] = backbone.features[0][0].weight[:, :min(3, config.in_channels), ...]
                        if config.in_channels > 3:
                            mean_weights = torch.mean(backbone.features[0][0].weight, dim=1)
                            first_conv.weight[:, 3:, ...] = mean_weights.unsqueeze(1)
            backbone.features[0][0] = first_conv
            
        # Split into stages
        self.stages = nn.ModuleList([
            backbone.features[0:2],   # stage 1: 32 channels
            backbone.features[2:3],   # stage 2: 24 channels
            backbone.features[3:4],   # stage 3: 40 channels
            backbone.features[4:6],   # stage 4: 112 channels
            backbone.features[6:]     # stage 5: 320 channels
        ])
        
        self.out_channels = [32, 24, 40, 112, 320]

class DecoderBlock(nn.Module):
    """Basic decoder block with skip connection support."""
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        # Handle initial upsampling conv
        self.up_conv = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        
        # After concatenating with skip connection
        combined_channels = (in_channels // 2) + skip_channels if skip_channels > 0 else in_channels // 2
        
        # Two convolutions to process combined features
        self.conv1 = nn.Conv2d(combined_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upconv instead of interpolate
        x = self.up_conv(x)
        
        # Concatenate skip connection if provided
        if skip is not None:
            # Handle case where sizes don't match exactly
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        # First conv + activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Second conv + activation    
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x