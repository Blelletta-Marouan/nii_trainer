from typing import Dict, List, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ..configs.config import ModelConfig

class EncoderFactory:
    """Factory for creating different types of encoders."""
    
    @staticmethod
    def create_encoder(config: ModelConfig) -> nn.Module:
        if config.encoder_type == "mobilenet_v2":
            return MobileNetV2Encoder(config)
        elif config.encoder_type == "resnet18":
            return ResNetEncoder(models.resnet18, config)
        elif config.encoder_type == "resnet50":
            return ResNetEncoder(models.resnet50, config)
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
        if config.in_channels != 3 and config.pretrained:
            # Initialize new channels with mean of pretrained weights
            with torch.no_grad():
                first_conv.weight[:, :3, ...] = backbone.features[0][0].weight
                if config.in_channels > 3:
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
    def __init__(self, model_fn: Type[nn.Module], config: ModelConfig):
        super().__init__()
        backbone = model_fn(pretrained=config.pretrained)
        
        # Modify first conv for arbitrary input channels
        if config.in_channels != 3:
            first_conv = nn.Conv2d(config.in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
            if config.pretrained:
                with torch.no_grad():
                    first_conv.weight[:, :3, ...] = backbone.conv1.weight
                    if config.in_channels > 3:
                        mean_weights = torch.mean(backbone.conv1.weight, dim=1)
                        first_conv.weight[:, 3:, ...] = mean_weights.unsqueeze(1)
            backbone.conv1 = first_conv
            
        # Create stages
        self.stages = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])
        
        # Set output channels based on ResNet variant
        if "18" in str(model_fn) or "34" in str(model_fn):
            self.out_channels = [64, 64, 128, 256, 512]
        else:  # ResNet50, 101, 152
            self.out_channels = [64, 256, 512, 1024, 2048]

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
                    first_conv.weight[:, :3, ...] = backbone.features[0][0].weight
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
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x