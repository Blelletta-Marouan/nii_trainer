"""
Tests for core functionality.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from nii_trainer.core.config import GlobalConfig, create_default_config, load_config
from nii_trainer.core.exceptions import NIITrainerError, ConfigurationError
from nii_trainer.core.registry import Registry, ENCODER_REGISTRY, DECODER_REGISTRY


class TestConfiguration:
    """Test configuration management."""
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()
        
        assert isinstance(config, GlobalConfig)
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        assert hasattr(config, 'evaluation')
    
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        # Valid config should pass
        sample_config.validate()
        
        # Invalid config should raise error
        sample_config.training.max_epochs = -1
        with pytest.raises(ConfigurationError):
            sample_config.validate()
    
    def test_config_serialization(self, sample_config):
        """Test configuration serialization."""
        # Test to_dict
        config_dict = sample_config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict
        
        # Test from_dict
        new_config = GlobalConfig.from_dict(config_dict)
        assert new_config.model.model_name == sample_config.model.model_name
    
    def test_config_yaml_operations(self, sample_config):
        """Test YAML save/load operations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to YAML
            sample_config.to_yaml(temp_path)
            assert Path(temp_path).exists()
            
            # Load from YAML
            loaded_config = load_config(temp_path)
            assert loaded_config.model.model_name == sample_config.model.model_name
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_hierarchical_access(self, sample_config):
        """Test hierarchical configuration access."""
        # Test nested attribute access
        assert hasattr(sample_config.model, 'model_name')
        assert hasattr(sample_config.training, 'max_epochs')
        
        # Test setting nested values
        sample_config.model.encoder = "resnet101"
        assert sample_config.model.encoder == "resnet101"


class TestExceptions:
    """Test custom exceptions."""
    
    def test_nii_trainer_error(self):
        """Test NIITrainerError exception."""
        with pytest.raises(NIITrainerError):
            raise NIITrainerError("Test error message")
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid configuration")
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        assert issubclass(ConfigurationError, NIITrainerError)


class TestRegistry:
    """Test registry system."""
    
    def test_registry_creation(self):
        """Test registry creation."""
        registry = Registry("test_registry")
        assert registry.name == "test_registry"
        assert len(registry._registry) == 0
    
    def test_registry_registration(self):
        """Test component registration."""
        registry = Registry("test_registry")
        
        class TestComponent:
            pass
        
        # Register component
        registry.register("test_component", TestComponent)
        
        # Check registration
        assert "test_component" in registry._registry
        assert registry.get("test_component") == TestComponent
    
    def test_registry_decorator(self):
        """Test registry decorator."""
        registry = Registry("test_registry")
        
        @registry.register("decorated_component")
        class DecoratedComponent:
            pass
        
        assert "decorated_component" in registry._registry
        assert registry.get("decorated_component") == DecoratedComponent
    
    def test_registry_error_handling(self):
        """Test registry error handling."""
        registry = Registry("test_registry")
        
        # Test getting non-existent component
        with pytest.raises(KeyError):
            registry.get("non_existent")
        
        # Test duplicate registration
        class TestComponent:
            pass
        
        registry.register("test", TestComponent)
        
        with pytest.raises(ValueError):
            registry.register("test", TestComponent)
    
    def test_global_registries(self):
        """Test global registry instances."""
        # Test that global registries exist
        assert ENCODER_REGISTRY is not None
        assert DECODER_REGISTRY is not None
        
        # Test registry names
        assert ENCODER_REGISTRY.name == "encoder"
        assert DECODER_REGISTRY.name == "decoder"
    
    def test_registry_list_components(self):
        """Test listing registered components."""
        registry = Registry("test_registry")
        
        class ComponentA:
            pass
        
        class ComponentB:
            pass
        
        registry.register("comp_a", ComponentA)
        registry.register("comp_b", ComponentB)
        
        components = registry.list_components()
        assert "comp_a" in components
        assert "comp_b" in components
        assert len(components) == 2


class TestDeviceManagement:
    """Test device management utilities."""
    
    def test_device_detection(self):
        """Test automatic device detection."""
        from nii_trainer.utils.device import get_device, get_system_info
        
        device = get_device()
        assert isinstance(device, torch.device)
        
        system_info = get_system_info()
        assert 'cuda_available' in system_info
        assert 'device_count' in system_info
    
    def test_device_selection(self):
        """Test manual device selection."""
        from nii_trainer.utils.device import get_device
        
        # Test CPU selection
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
        
        # Test CUDA selection (if available)
        if torch.cuda.is_available():
            cuda_device = get_device("cuda")
            assert cuda_device.type == "cuda"


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_setup(self):
        """Test logger setup."""
        from nii_trainer.utils.logging import setup_logging, get_logger
        
        # Test basic setup
        logger = setup_logging()
        assert logger is not None
        
        # Test getting logger
        named_logger = get_logger("test_logger")
        assert named_logger.name == "test_logger"
    
    def test_logger_manager(self):
        """Test logger manager."""
        from nii_trainer.utils.logging import LoggerManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoggerManager(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_tensorboard=False,  # Disable for testing
                use_wandb=False
            )
            
            assert manager.experiment_name == "test_experiment"
            assert manager.logger is not None
            
            # Test logging metrics
            metrics = {"loss": 0.5, "accuracy": 0.85}
            manager.log_metrics(metrics, step=1)
            
            manager.close()


class TestMemoryManagement:
    """Test memory management utilities."""
    
    def test_memory_optimization(self):
        """Test memory optimization utilities."""
        from nii_trainer.utils.memory import optimize_memory_usage, clear_cache
        
        # Test memory optimization (should not raise errors)
        optimize_memory_usage()
        clear_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        from nii_trainer.utils.memory import get_gpu_memory_usage, clear_cache
        
        # Test getting GPU memory usage
        memory_info = get_gpu_memory_usage()
        assert 'total' in memory_info
        assert 'allocated' in memory_info
        assert 'cached' in memory_info
        
        # Test clearing cache
        clear_cache()


class TestIOUtilities:
    """Test I/O utilities."""
    
    def test_checkpoint_operations(self, sample_config, mock_model):
        """Test checkpoint save/load operations."""
        from nii_trainer.utils.io import save_checkpoint, load_checkpoint
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Create mock optimizer
            optimizer = torch.optim.Adam(mock_model.parameters())
            
            # Save checkpoint
            save_checkpoint(
                model=mock_model,
                optimizer=optimizer,
                epoch=10,
                loss=0.5,
                metrics={'dice': 0.85},
                filepath=checkpoint_path
            )
            
            assert Path(checkpoint_path).exists()
            
            # Load checkpoint
            checkpoint_data = load_checkpoint(
                filepath=checkpoint_path,
                model=mock_model,
                optimizer=optimizer
            )
            
            assert checkpoint_data['epoch'] == 10
            assert checkpoint_data['loss'] == 0.5
            assert checkpoint_data['metrics']['dice'] == 0.85
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_prediction_saving(self):
        """Test prediction saving utilities."""
        from nii_trainer.utils.io import save_predictions
        
        predictions = {
            'prediction': torch.randn(1, 1, 64, 64),
            'input_path': 'test_image.nii.gz'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            output_path = f.name
        
        try:
            save_predictions(predictions, output_path)
            assert Path(output_path).exists()
            
        finally:
            Path(output_path).unlink(missing_ok=True)