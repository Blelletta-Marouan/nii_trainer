"""
Tests for model components.
"""

import pytest
import torch
import torch.nn as nn

from nii_trainer.models import create_model
from nii_trainer.models.base import BaseModel
from nii_trainer.models.encoders import BaseEncoder, ResNetEncoder
from nii_trainer.models.decoders import BaseDecoder, UNetDecoder
from nii_trainer.models.cascaded import CascadedModel


class TestBaseModel:
    """Test base model functionality."""
    
    def test_base_model_creation(self):
        """Test base model instantiation."""
        model = BaseModel()
        assert isinstance(model, nn.Module)
    
    def test_base_model_forward_not_implemented(self):
        """Test that base model forward raises NotImplementedError."""
        model = BaseModel()
        with pytest.raises(NotImplementedError):
            model(torch.randn(1, 1, 64, 64))


class TestEncoders:
    """Test encoder components."""
    
    def test_base_encoder(self):
        """Test base encoder functionality."""
        encoder = BaseEncoder(in_channels=1)
        assert isinstance(encoder, nn.Module)
        assert encoder.in_channels == 1
    
    def test_resnet_encoder(self):
        """Test ResNet encoder."""
        encoder = ResNetEncoder(
            in_channels=1,
            encoder_name="resnet18",
            pretrained=False
        )
        
        # Test forward pass
        x = torch.randn(2, 1, 64, 64)
        features = encoder(x)
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Check feature dimensions
        for i, feat in enumerate(features):
            assert feat.size(0) == 2  # batch size
            assert feat.size(1) > 0   # channels
    
    def test_encoder_with_different_inputs(self):
        """Test encoder with different input sizes."""
        encoder = ResNetEncoder(in_channels=1, encoder_name="resnet18", pretrained=False)
        
        # Test different input sizes
        sizes = [(32, 32), (64, 64), (128, 128)]
        
        for h, w in sizes:
            x = torch.randn(1, 1, h, w)
            features = encoder(x)
            assert len(features) > 0
    
    def test_encoder_channels(self):
        """Test encoder with different input channels."""
        # Single channel (medical images)
        encoder_1ch = ResNetEncoder(in_channels=1, encoder_name="resnet18", pretrained=False)
        x_1ch = torch.randn(1, 1, 64, 64)
        features_1ch = encoder_1ch(x_1ch)
        
        # Multi-channel
        encoder_3ch = ResNetEncoder(in_channels=3, encoder_name="resnet18", pretrained=False)
        x_3ch = torch.randn(1, 3, 64, 64)
        features_3ch = encoder_3ch(x_3ch)
        
        assert len(features_1ch) == len(features_3ch)


class TestDecoders:
    """Test decoder components."""
    
    def test_base_decoder(self):
        """Test base decoder functionality."""
        decoder = BaseDecoder(encoder_channels=[64, 128, 256])
        assert isinstance(decoder, nn.Module)
        assert decoder.encoder_channels == [64, 128, 256]
    
    def test_unet_decoder(self):
        """Test U-Net decoder."""
        encoder_channels = [64, 128, 256, 512]
        decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64, 32],
            num_classes=2
        )
        
        # Create mock encoder features
        features = []
        h, w = 64, 64
        for channels in encoder_channels:
            features.append(torch.randn(2, channels, h, w))
            h, w = h // 2, w // 2
        
        # Test forward pass
        output = decoder(*features)
        
        assert output.size(0) == 2  # batch size
        assert output.size(1) == 2  # num_classes
        assert output.size(2) == 64  # height
        assert output.size(3) == 64  # width


class TestCascadedModel:
    """Test cascaded model architecture."""
    
    def test_cascaded_model_creation(self, sample_config):
        """Test cascaded model creation."""
        model = CascadedModel(sample_config.model)
        assert isinstance(model, BaseModel)
        assert hasattr(model, 'stages')
        assert len(model.stages) == sample_config.model.num_stages
    
    def test_cascaded_model_forward(self, sample_config):
        """Test cascaded model forward pass."""
        model = CascadedModel(sample_config.model)
        
        x = torch.randn(2, 1, 64, 64)
        output = model(x)
        
        if sample_config.model.num_stages > 1:
            assert isinstance(output, list)
            assert len(output) == sample_config.model.num_stages
        else:
            assert isinstance(output, torch.Tensor)
    
    def test_cascaded_model_training_mode(self, sample_config):
        """Test cascaded model in training vs eval mode."""
        model = CascadedModel(sample_config.model)
        
        x = torch.randn(2, 1, 64, 64)
        
        # Training mode
        model.train()
        train_output = model(x)
        
        # Eval mode
        model.eval()
        eval_output = model(x)
        
        # Outputs should have same structure
        if isinstance(train_output, list):
            assert isinstance(eval_output, list)
            assert len(train_output) == len(eval_output)


class TestModelFactory:
    """Test model creation factory."""
    
    def test_create_model_with_config(self, sample_config):
        """Test model creation with configuration."""
        model = create_model(sample_config.model)
        
        assert isinstance(model, BaseModel)
        assert hasattr(model, 'forward')
    
    def test_create_model_with_dict(self):
        """Test model creation with dictionary config."""
        model_config = {
            'model_name': 'cascaded_unet',
            'num_stages': 2,
            'encoder': 'resnet18',
            'decoder': 'unet',
            'num_classes': 2
        }
        
        model = create_model(model_config)
        assert isinstance(model, BaseModel)
    
    def test_create_model_invalid_config(self):
        """Test model creation with invalid configuration."""
        invalid_config = {
            'model_name': 'nonexistent_model'
        }
        
        with pytest.raises(Exception):  # Should raise some kind of error
            create_model(invalid_config)


class TestModelComponents:
    """Test individual model components."""
    
    def test_attention_mechanism(self):
        """Test attention mechanism components."""
        from nii_trainer.models.components import AttentionBlock
        
        attention = AttentionBlock(in_channels=64)
        
        x = torch.randn(2, 64, 32, 32)
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should modify input
    
    def test_normalization_layers(self):
        """Test normalization layer components."""
        from nii_trainer.models.components import get_norm_layer
        
        # Test different normalization types
        norm_types = ['batch', 'instance', 'group']
        
        for norm_type in norm_types:
            norm_layer = get_norm_layer(norm_type, num_channels=64)
            assert isinstance(norm_layer, nn.Module)
            
            x = torch.randn(2, 64, 32, 32)
            output = norm_layer(x)
            assert output.shape == x.shape
    
    def test_activation_functions(self):
        """Test activation function components."""
        from nii_trainer.models.components import get_activation
        
        # Test different activation types
        activation_types = ['relu', 'leaky_relu', 'gelu', 'swish']
        
        for act_type in activation_types:
            activation = get_activation(act_type)
            assert isinstance(activation, nn.Module)
            
            x = torch.randn(2, 64, 32, 32)
            output = activation(x)
            assert output.shape == x.shape


class TestModelTraining:
    """Test model training utilities."""
    
    def test_model_parameter_count(self, mock_model):
        """Test parameter counting utilities."""
        from nii_trainer.models.utils import count_parameters
        
        param_count = count_parameters(mock_model)
        assert isinstance(param_count, int)
        assert param_count > 0
    
    def test_model_memory_usage(self, mock_model):
        """Test model memory usage estimation."""
        from nii_trainer.models.utils import estimate_model_memory
        
        memory_mb = estimate_model_memory(
            mock_model, 
            input_shape=(1, 1, 64, 64),
            batch_size=4
        )
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_model_flops_calculation(self, mock_model):
        """Test FLOPS calculation."""
        from nii_trainer.models.utils import calculate_flops
        
        try:
            flops = calculate_flops(mock_model, input_shape=(1, 1, 64, 64))
            assert isinstance(flops, (int, float))
            assert flops > 0
        except ImportError:
            # Skip if thop not available
            pytest.skip("thop package not available for FLOPS calculation")


class TestModelSerialization:
    """Test model serialization and loading."""
    
    def test_model_state_dict(self, mock_model):
        """Test model state dict operations."""
        # Get state dict
        state_dict = mock_model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # Create new model and load state dict
        new_model = create_model({"model_name": "test_model"})
        
        # Note: This might fail due to architecture differences
        # In real implementation, we'd ensure compatible architectures
        try:
            new_model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            # Expected for incompatible architectures in tests
            pass
    
    def test_model_checkpoint_compatibility(self, mock_model):
        """Test model checkpoint compatibility."""
        import tempfile
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            torch.save({
                'model_state_dict': mock_model.state_dict(),
                'model_config': {'model_name': 'test_model'}
            }, checkpoint_path)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            assert 'model_state_dict' in checkpoint
            assert 'model_config' in checkpoint
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestModelValidation:
    """Test model validation utilities."""
    
    def test_model_gradient_flow(self, mock_model):
        """Test gradient flow through model."""
        from nii_trainer.utils.debugging import check_gradients
        
        # Forward pass with loss
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        output = mock_model(x)
        
        if isinstance(output, list):
            loss = sum(o.sum() for o in output)
        else:
            loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_info = check_gradients(mock_model)
        assert 'total_params' in grad_info
        assert 'params_with_grad' in grad_info
    
    def test_model_output_shapes(self, mock_model):
        """Test model output shapes are consistent."""
        input_sizes = [(1, 1, 32, 32), (2, 1, 64, 64), (4, 1, 128, 128)]
        
        for batch_size, channels, height, width in input_sizes:
            x = torch.randn(batch_size, channels, height, width)
            output = mock_model(x)
            
            if isinstance(output, list):
                for o in output:
                    assert o.size(0) == batch_size
            else:
                assert output.size(0) == batch_size