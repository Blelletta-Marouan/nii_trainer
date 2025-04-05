import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Tuple
from .base import EncoderFactory, DecoderBlock
from ..configs.config import CascadeConfig, StageConfig, ModelConfig
from ..utils.logging_utils import setup_logger

class CascadeStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        initial_features: int,
        stage_config: StageConfig,
        pretrained: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.config = stage_config
        self.logger = logger
        self.is_binary = stage_config.is_binary
        
        if self.logger:
            self.logger.info(f"Initializing cascade stage:")
            self.logger.info(f"  Input channels: {in_channels}")
            self.logger.info(f"  Initial features: {initial_features}")
            self.logger.info(f"  Encoder type: {stage_config.encoder_type}")
            self.logger.info(f"  Num layers: {stage_config.num_layers}")
            self.logger.info(f"  Is binary: {stage_config.is_binary}")
        
        # Create encoder
        self.encoder = self._create_encoder(
            in_channels,
            initial_features,
            stage_config,
            pretrained
        )
        
        # Get encoder output channels
        encoder_channels = self.encoder.out_channels
        
        if self.logger:
            self.logger.info(f"  Encoder channels: {encoder_channels}")
        
        # Create decoder stages
        self.decoder = self._create_decoder(
            encoder_channels,
            stage_config.num_layers,
            stage_config.dropout_rate
        )

    def _create_encoder(
        self,
        in_channels: int,
        initial_features: int,
        config: StageConfig,
        pretrained: bool
    ) -> nn.Module:
        """Create encoder with proper configuration."""
        encoder_config = ModelConfig(
            encoder_type=config.encoder_type,
            in_channels=in_channels,
            initial_features=initial_features,
            num_layers=config.num_layers,
            pretrained=pretrained
        )
        return EncoderFactory.create_encoder(encoder_config)
        
    def _create_decoder(
        self,
        encoder_channels: List[int],
        num_layers: int,
        dropout_rate: float
    ) -> nn.ModuleList:
        """Create decoder with specified number of layers."""
        if num_layers < 2 or num_layers > 7:
            raise ValueError(f"num_layers must be between 2 and 7, got {num_layers}")
            
        decoder = nn.ModuleList()
        reversed_channels = encoder_channels[::-1]
        
        # Custom handling for EfficientNet
        if self.config.encoder_type == "efficientnet":
            # Define all possible decoder channels for EfficientNet (from deepest to shallowest)
            all_decoder_channels = [
                (320, 112),   # Block 1: in=320, out=112 (deepest)
                (112, 40),    # Block 2: in=112, out=40
                (40, 24),     # Block 3: in=40, out=24
                (24, 16),     # Block 4: in=24, out=16
                (16, 8),      # Block 5: in=16, out=8 (if needed)
                (8, 4),       # Block 6: in=8, out=4 (if needed)
                (4, 2)        # Block 7: in=4, out=2 (shallowest, if needed)
            ]
            
            # Select the appropriate number of decoder blocks based on num_layers
            # Subtract 1 since the final 1x1 convolution is not part of these blocks
            decoder_channels = all_decoder_channels[:min(num_layers-1, len(all_decoder_channels))]
            
            if self.logger:
                self.logger.debug(f"EfficientNet decoder using {len(decoder_channels)} blocks with channels: {decoder_channels}")
            
            # Add decoder blocks with selected channel dimensions
            for i, (in_ch, out_ch) in enumerate(decoder_channels):
                # Skip connection index matches the current decoder level
                skip_idx = i + 1
                skip_ch = reversed_channels[skip_idx] if skip_idx < len(reversed_channels) and self.config.skip_connections else 0
                
                # Create specialized ConvTranspose-based upsampling block
                decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                )
            
            # Final convolution layer to get desired output channels
            out_channels = 1 if self.is_binary else len(self.config.input_classes)
            # Use the output channels from the last decoder block as input channels
            last_output_ch = decoder_channels[-1][1] if decoder_channels else 320
            
            if self.logger:
                self.logger.debug(f"EfficientNet final conv input channels: {last_output_ch}, output channels: {out_channels}")
                
            decoder.append(nn.Conv2d(last_output_ch, out_channels, kernel_size=1))
            
        else:
            # Standard UNet decoder for non-EfficientNet encoders
            # Calculate how many decoder blocks to create based on num_layers
            decoder_depth = min(num_layers - 1, len(reversed_channels) - 1)
            
            for i in range(decoder_depth):
                in_ch = reversed_channels[i]
                out_ch = reversed_channels[i + 1] if i < len(reversed_channels) - 1 else reversed_channels[-1] // 2
                skip_ch = reversed_channels[i + 1] if i < len(reversed_channels) - 1 and self.config.skip_connections else 0
                
                if self.logger:
                    self.logger.debug(f"Decoder block {i}: in_ch={in_ch}, skip_ch={skip_ch}, out_ch={out_ch}")
                
                decoder.append(
                    DecoderBlock(
                        in_channels=in_ch,
                        skip_channels=skip_ch,
                        out_channels=out_ch,
                        dropout_rate=dropout_rate
                    )
                )
            
            # Add additional decoder blocks if needed (more decoder than encoder layers)
            last_out_ch = reversed_channels[-1] // 2 if decoder_depth > 0 else reversed_channels[0] // 2
            for i in range(decoder_depth, num_layers - 1):
                in_ch = last_out_ch
                out_ch = in_ch // 2
                last_out_ch = out_ch
                
                if self.logger:
                    self.logger.debug(f"Additional decoder block {i}: in_ch={in_ch}, out_ch={out_ch}")
                
                decoder.append(
                    DecoderBlock(
                        in_channels=in_ch,
                        skip_channels=0,  # No skip connections for additional layers
                        out_channels=out_ch,
                        dropout_rate=dropout_rate
                    )
                )
            
            # Final convolution to get desired output channels
            out_channels = 1 if self.is_binary else len(self.config.input_classes)
            
            if self.logger:
                self.logger.debug(f"Final conv input channels: {last_out_ch}, output channels: {out_channels}")
                
            decoder.append(nn.Conv2d(last_out_ch, out_channels, kernel_size=1))
        
        return decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single stage."""
        # Store original input size for later resizing
        input_size = x.shape[-2:]
        
        # Encode
        enc_features = self.encoder(x)
        
        # Get encoder feature dimensions
        encoder_channels = [f.size(1) for f in enc_features]
        
        # Decode - custom implementation for EfficientNet
        if self.config.encoder_type == "efficientnet":
            # Use bottleneck features as starting point
            features = enc_features[::-1]  # Reverse features for decoder
            bottleneck = features[0]  # [B, 320, H/32, W/32]
            
            # Apply each decoder block manually with appropriate channel handling
            for i, block in enumerate(self.decoder[:-1]):
                # Get appropriate skip connection
                skip_idx = i + 1
                skip = features[skip_idx] if skip_idx < len(features) and self.config.skip_connections else None
                
                # Create projection if channels don't match expected input
                expected_in_channels = block[0].in_channels
                if bottleneck.size(1) != expected_in_channels:
                    # Create dynamic projection layer
                    proj = nn.Conv2d(
                        bottleneck.size(1), 
                        expected_in_channels, 
                        kernel_size=1
                    ).to(bottleneck.device)
                    
                    # Initialize with identity where possible
                    with torch.no_grad():
                        nn.init.zeros_(proj.weight)
                        nn.init.zeros_(proj.bias)
                        min_channels = min(bottleneck.size(1), expected_in_channels)
                        for j in range(min_channels):
                            proj.weight[j, j, 0, 0] = 1.0
                    
                    # Apply projection
                    bottleneck = proj(bottleneck)
                
                # Apply transpose convolution upsampling
                upsampled = block[0](bottleneck)  # ConvTranspose
                upsampled = block[1](upsampled)   # BatchNorm
                upsampled = block[2](upsampled)   # ReLU
                
                # Concatenate with skip connection if available
                if skip is not None:
                    # Ensure same spatial dimensions
                    if upsampled.shape[2:] != skip.shape[2:]:
                        upsampled = F.interpolate(
                            upsampled,
                            size=skip.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Concatenate
                    combined = torch.cat([upsampled, skip], dim=1)
                    
                    # Apply remaining convolution operations
                    bottleneck = block[3](combined)  # Conv
                    bottleneck = block[4](bottleneck)  # BatchNorm
                    bottleneck = block[5](bottleneck)  # ReLU
                else:
                    bottleneck = upsampled
            
            # Apply final 1x1 convolution to get output channels
            output = self.decoder[-1](bottleneck)
            
        else:
            # Standard UNet decoder with DecoderBlocks
            dec_features = enc_features[::-1]  # Reverse for decoder
            out = dec_features[0]
            
            for i, dec_block in enumerate(self.decoder[:-1]):
                skip = dec_features[i + 1] if i + 1 < len(dec_features) and self.config.skip_connections else None
                
                # Debug channel dimensions
                if self.logger and self.logger.level <= logging.DEBUG:
                    self.logger.debug(f"Decoder block {i}: out shape={out.shape}, " + 
                                     (f"skip shape={skip.shape}" if skip is not None else "no skip"))
                
                # Check if we need to add a channel adapter
                if isinstance(dec_block, DecoderBlock) and out.size(1) != dec_block.up_conv.in_channels:
                    if self.logger:
                        self.logger.debug(f"Adding channel adapter from {out.size(1)} to {dec_block.up_conv.in_channels}")
                    adapter = nn.Conv2d(out.size(1), dec_block.up_conv.in_channels, kernel_size=1).to(out.device)
                    # Initialize with identity mapping where possible
                    with torch.no_grad():
                        nn.init.zeros_(adapter.weight)
                        nn.init.zeros_(adapter.bias)
                        min_channels = min(out.size(1), dec_block.up_conv.in_channels)
                        for j in range(min_channels):
                            adapter.weight[j, j, 0, 0] = 1.0
                    out = adapter(out)
                
                # Apply decoder block
                try:
                    out = dec_block(out, skip)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in decoder block {i}: {str(e)}")
                    raise
            
            # Apply final 1x1 convolution with channel adapter if needed
            final_conv = self.decoder[-1]
            if isinstance(final_conv, nn.Conv2d) and out.size(1) != final_conv.in_channels:
                if self.logger:
                    self.logger.debug(f"Adding final channel adapter from {out.size(1)} to {final_conv.in_channels}")
                adapter = nn.Conv2d(out.size(1), final_conv.in_channels, kernel_size=1).to(out.device)
                # Initialize with identity mapping where possible
                with torch.no_grad():
                    nn.init.zeros_(adapter.weight)
                    nn.init.zeros_(adapter.bias)
                    min_channels = min(out.size(1), final_conv.in_channels)
                    for j in range(min_channels):
                        adapter.weight[j, j, 0, 0] = 1.0
                out = adapter(out)
            
            # Apply final 1x1 convolution
            output = final_conv(out)
        
        # Ensure output has the same spatial dimensions as input
        if output.shape[-2:] != input_size:
            if self.logger:
                self.logger.debug(f"Resizing output from {output.shape[-2:]} to {input_size}")
            output = F.interpolate(
                output,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        return output

class FlexibleCascadedUNet(nn.Module):
    """
    A flexible cascaded UNet that supports:
    - Binary segmentation in first stage
    - Arbitrary number of cascade stages
    - Different encoder backbones per stage
    - Configurable number of layers per stage
    - Flexible class handling per stage
    """
    def __init__(self, config: CascadeConfig, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.config = config
        
        # Setup logger if not provided
        if logger is None:
            logger = setup_logger("cascaded_unet")
        self.logger = logger
        
        self.logger.info("Initializing FlexibleCascadedUNet:")
        self.logger.info(f"Input channels: {config.in_channels}")
        self.logger.info(f"Initial features: {config.initial_features}")
        self.logger.info(f"Feature growth: {config.feature_growth}")
        self.logger.info(f"Number of stages: {len(config.stages)}")
        
        self.stages = nn.ModuleList()
        
        # Create each stage
        for stage_idx, stage_config in enumerate(config.stages):
            self.logger.info(f"\nInitializing Stage {stage_idx + 1}:")
            # Calculate input channels for this stage
            if stage_idx == 0:
                stage_in_channels = config.in_channels
            else:
                # Input channels include original input plus all previous stage outputs
                stage_in_channels = config.in_channels + len(stage_config.input_classes)
                
            # Create stage
            stage = CascadeStage(
                in_channels=stage_in_channels,
                initial_features=int(config.initial_features * (config.feature_growth ** stage_idx)),
                stage_config=stage_config,
                pretrained=config.pretrained,
                logger=self.logger
            )
            self.stages.append(stage)
            
        self.logger.info("\nModel initialization complete")
        
    def forward(self, x, active_stages=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, C, H, W)
            active_stages: Optional list of stage indices to compute (for curriculum learning)
            
        Returns:
            List of predictions from each stage
        """
        outputs = []
        batch_size = x.shape[0]
        image_size = x.shape[2:]

        # Determine which stages to run
        all_stages = range(len(self.stages))
        stages_to_run = active_stages if active_stages is not None else all_stages
        
        # Create empty placeholders for all stages to maintain output structure
        placeholders = [torch.zeros((batch_size, 1, *image_size), device=x.device) for _ in all_stages]
        outputs = placeholders.copy()  # Start with placeholders for all stages
        
        # Previous stage prediction cache for memory efficiency
        prev_stage_pred = None
        
        # Process only active stages
        for stage_idx in all_stages:
            if stage_idx not in stages_to_run:
                continue
                
            # Free up memory from previous inactive stages
            if prev_stage_pred is not None and stage_idx - 1 not in stages_to_run:
                prev_stage_pred = None
                torch.cuda.empty_cache()  # Release unused GPU memory
            
            # Get stage input
            if stage_idx == 0:
                # First stage just takes original input
                stage_input = x
            else:
                # Update prev_stage_pred if needed
                if prev_stage_pred is None:
                    prev_stage_pred = (torch.sigmoid(outputs[stage_idx - 1]) > 0.5).float()
                # Concatenate original input with previous stage mask
                stage_input = torch.cat([x, prev_stage_pred], dim=1)
            
            # Get stage output
            with torch.set_grad_enabled(stage_idx in stages_to_run):  # Only compute gradients for active stages
                stage_output = self.stages[stage_idx](stage_input)
                outputs[stage_idx] = stage_output
                
                # Cache prediction for next stage if needed
                if stage_idx + 1 in stages_to_run:
                    prev_stage_pred = (torch.sigmoid(stage_output) > 0.5).float()
                else:
                    prev_stage_pred = None
        
        return outputs
    
    def get_predictions(self, outputs: List[torch.Tensor], thresholds: Optional[List[float]] = None) -> torch.Tensor:
        """
        Convert model outputs to final predictions using thresholds.
        First stage uses binary threshold, subsequent stages use class-specific thresholds.
        Returns tensor of shape [B, H, W] with class indices.
        
        Args:
            outputs: List of tensors from model forward pass
            thresholds: Optional list of thresholds for each stage
                        If None, uses default thresholds from stage configs
        """
        if thresholds is None:
            thresholds = [stage.config.threshold for stage in self.stages]
            
        # Ensure we have a threshold for each output
        if len(thresholds) != len(outputs):
            thresholds = thresholds[:len(outputs)]
            
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
        
        # Subsequent stages: class-specific segmentation within foreground
        for i, (out, thresh) in enumerate(list(zip(outputs[1:], thresholds[1:]))):
            prob = torch.sigmoid(out)
            mask = (prob > thresh).squeeze(1)
            # Only apply within the foreground region from first stage
            final_mask[mask & binary_mask] = i + 2
            
        return final_mask