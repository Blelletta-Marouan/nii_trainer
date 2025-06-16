"""
Base model classes for NII-Trainer.

This module provides abstract base classes and common functionality
for all models in the NII-Trainer framework.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union

from ..core.exceptions import ModelError
from ..core.registry import register_model


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count, architecture details."""
        pass
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specified layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """Unfreeze specified layers."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    def get_layer_names(self) -> List[str]:
        """Get names of all layers in the model."""
        return [name for name, _ in self.named_modules()]


class BaseEncoder(BaseModel):
    """Base class for encoder architectures."""
    
    def __init__(self, in_channels: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.features = []  # Store feature maps at different resolutions
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Returns:
            List of feature tensors at different scales
        """
        pass


class BaseDecoder(BaseModel):
    """Base class for decoder architectures."""
    
    def __init__(self, out_channels: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass taking multi-scale features.
        
        Args:
            features: List of feature tensors from encoder
            
        Returns:
            Output tensor
        """
        pass


class BaseCascadeModel(BaseModel):
    """Base class for cascaded models."""
    
    def __init__(self, num_stages: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
    
    @abstractmethod
    def forward_stage(self, x: torch.Tensor, stage_idx: int, 
                     previous_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass for a single stage."""
        pass
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all stages.
        
        Returns:
            List of outputs from each stage
        """
        outputs = []
        previous_outputs = None
        
        for stage_idx in range(self.num_stages):
            stage_output = self.forward_stage(x, stage_idx, previous_outputs)
            outputs.append(stage_output)
            previous_outputs = outputs.copy()
        
        return outputs
    
    def forward_single_stage(self, x: torch.Tensor, stage_idx: int,
                           previous_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass for a single stage only."""
        return self.forward_stage(x, stage_idx, previous_outputs)


class StageModule(nn.Module):
    """Individual stage in a cascaded model."""
    
    def __init__(self, 
                 stage_id: int,
                 encoder: BaseEncoder,
                 decoder: BaseDecoder,
                 num_classes: int,
                 depends_on: List[int] = None,
                 fusion_strategy: str = 'concatenate'):
        """
        Initialize stage module.
        
        Args:
            stage_id: Unique identifier for this stage
            encoder: Encoder architecture
            decoder: Decoder architecture  
            num_classes: Number of output classes
            depends_on: List of previous stage IDs this stage depends on
            fusion_strategy: How to fuse inputs from previous stages
        """
        super().__init__()
        self.stage_id = stage_id
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.depends_on = depends_on or []
        self.fusion_strategy = fusion_strategy
        
        # Output projection
        self.output_proj = nn.Conv2d(decoder.out_channels, num_classes, kernel_size=1)
        
        # Fusion layers if needed
        if self.depends_on:
            self._setup_fusion_layers()
    
    def _setup_fusion_layers(self):
        """Setup layers for fusing inputs from previous stages."""
        if self.fusion_strategy == 'concatenate':
            # Additional channels from previous stages
            extra_channels = len(self.depends_on) * self.num_classes
            self.input_fusion = nn.Conv2d(
                self.encoder.in_channels + extra_channels,
                self.encoder.in_channels,
                kernel_size=3, padding=1
            )
        elif self.fusion_strategy == 'add':
            # Ensure compatible dimensions
            self.input_fusion = nn.Conv2d(
                self.num_classes, self.encoder.in_channels, kernel_size=1
            )
        elif self.fusion_strategy == 'attention':
            self.attention_fusion = AttentionFusion(
                self.encoder.in_channels, len(self.depends_on)
            )
    
    def forward(self, x: torch.Tensor,
                previous_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through this stage."""
        # Prepare input with fusion from previous stages
        stage_input = self._fuse_inputs(x, previous_outputs)
        
        # Encode
        features = self.encoder(stage_input)
        
        # Decode
        decoded = self.decoder(features)
        
        # Output projection
        output = self.output_proj(decoded)
        
        return output
    
    def _fuse_inputs(self, x: torch.Tensor,
                    previous_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Fuse current input with outputs from previous stages."""
        if not self.depends_on or previous_outputs is None:
            return x
        
        # Get relevant previous outputs
        relevant_outputs = [previous_outputs[idx] for idx in self.depends_on if idx < len(previous_outputs)]
        
        if not relevant_outputs:
            return x
        
        if self.fusion_strategy == 'concatenate':
            # Concatenate along channel dimension
            fused_input = torch.cat([x] + relevant_outputs, dim=1)
            return self.input_fusion(fused_input)
        
        elif self.fusion_strategy == 'add':
            # Add previous outputs to current input
            fused = x
            for prev_out in relevant_outputs:
                # Project to input space if needed
                projected = self.input_fusion(prev_out)
                fused = fused + projected
            return fused
        
        elif self.fusion_strategy == 'attention':
            return self.attention_fusion(x, relevant_outputs)
        
        else:
            raise ModelError(f"Unknown fusion strategy: {self.fusion_strategy}")


class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple inputs."""
    
    def __init__(self, in_channels: int, num_inputs: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        
        # Attention weights computation
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * (num_inputs + 1), in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_inputs + 1, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Input projection layers
        self.input_projs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(num_inputs)
        ])
    
    def forward(self, primary_input: torch.Tensor,
                auxiliary_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Apply attention-based fusion."""
        # Project auxiliary inputs
        proj_inputs = [proj(aux_inp) for proj, aux_inp in zip(self.input_projs, auxiliary_inputs)]
        
        # Concatenate all inputs for attention computation
        all_inputs = [primary_input] + proj_inputs
        concat_inputs = torch.cat(all_inputs, dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(concat_inputs)
        
        # Apply attention weights
        fused_output = torch.zeros_like(primary_input)
        for i, inp in enumerate(all_inputs):
            weight = attention_weights[:, i:i+1, :, :]
            fused_output += weight * inp
        
        return fused_output


def calculate_receptive_field(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """Calculate the receptive field of a model."""
    # This is a simplified calculation - would need more sophisticated implementation
    # for complex architectures
    receptive_field = 1
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            receptive_field = (receptive_field - 1) * stride + kernel_size
    
    return receptive_field


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
    """Get comprehensive model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'input_size': input_size,
        'receptive_field': calculate_receptive_field(model, input_size)
    }
    
    return summary