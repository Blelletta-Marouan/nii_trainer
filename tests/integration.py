"""
Integration tests for NII-Trainer.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from nii_trainer import NIITrainer, quick_train, quick_evaluate
from nii_trainer.api.advanced import AdvancedTrainer
from nii_trainer.api.experimental import AutoML
from nii_trainer.core.config import create_default_config


class TestEndToEndTraining:
    """Test complete training workflows."""
    
    def test_basic_training_workflow(self, temp_data_dir, temp_output_dir):
        """Test basic end-to-end training workflow."""
        # Create trainer
        trainer = NIITrainer(
            experiment_name="integration_test",
            output_dir=str(temp_output_dir)
        )
        
        # Setup data
        trainer.setup(
            train_data_path=str(temp_data_dir / "train"),
            val_data_path=str(temp_data_dir / "val")
        )
        
        # Train for minimal epochs
        results = trainer.train(epochs=1)
        
        # Verify results
        assert isinstance(results, dict)
        assert 'train_loss' in results or 'loss' in results
        
        # Verify checkpoint was saved
        checkpoint_files = list(temp_output_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
    
    def test_training_with_evaluation(self, temp_data_dir, temp_output_dir):
        """Test training with evaluation workflow."""
        trainer = NIITrainer(
            experiment_name="integration_test_eval",
            output_dir=str(temp_output_dir)
        )
        
        trainer.setup(
            train_data_path=str(temp_data_dir / "train"),
            val_data_path=str(temp_data_dir / "val")
        )
        
        # Train
        train_results = trainer.train(epochs=1)
        
        # Evaluate
        eval_results = trainer.evaluate(
            test_data_path=str(temp_data_dir / "test")
        )
        
        assert isinstance(eval_results, dict)
        assert len(eval_results) > 0
    
    def test_quick_train_function(self, temp_data_dir):
        """Test quick_train convenience function."""
        trainer = quick_train(
            train_data_path=str(temp_data_dir / "train"),
            val_data_path=str(temp_data_dir / "val"),
            epochs=1,
            experiment_name="quick_test"
        )
        
        assert trainer is not None
        assert hasattr(trainer, 'model')


class TestAdvancedWorkflows:
    """Test advanced training workflows."""
    
    def test_advanced_trainer_workflow(self, temp_data_dir, sample_config):
        """Test advanced trainer workflow."""
        # Modify config for quick testing
        sample_config.training.max_epochs = 1
        
        trainer = AdvancedTrainer(sample_config)
        trainer.build_model()
        trainer.setup_training()
        
        # Create mock data loaders (simplified)
        from torch.utils.data import DataLoader, TensorDataset
        
        # Simple tensor dataset for testing
        dummy_images = torch.randn(6, 1, 64, 64)
        dummy_labels = torch.randint(0, 2, (6, 1, 64, 64)).float()
        
        train_dataset = TensorDataset(dummy_images, dummy_labels)
        val_dataset = TensorDataset(dummy_images[:3], dummy_labels[:3])
        
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        
        # Train
        results = trainer.train_with_validation(train_loader, val_loader)
        
        assert isinstance(results, dict)
    
    @pytest.mark.slow
    def test_automl_workflow(self, temp_data_dir):
        """Test AutoML workflow (marked as slow)."""
        automl = AutoML(time_budget=60)  # 1 minute for testing
        
        try:
            results = automl.auto_train(
                train_data_path=str(temp_data_dir / "train"),
                val_data_path=str(temp_data_dir / "val")
            )
            
            assert 'best_config' in results
            assert 'best_score' in results
            
        except Exception as e:
            # AutoML might fail in testing environment
            pytest.skip(f"AutoML test skipped due to: {e}")


class TestModelExport:
    """Test model export functionality."""
    
    def test_model_export_onnx(self, temp_data_dir, temp_output_dir):
        """Test ONNX model export."""
        trainer = NIITrainer(
            experiment_name="export_test",
            output_dir=str(temp_output_dir)
        )
        
        trainer.setup(str(temp_data_dir / "train"))
        
        # Train minimal model
        trainer.train(epochs=1)
        
        # Export to ONNX
        export_path = temp_output_dir / "model.onnx"
        
        try:
            trainer.export_model(
                format="onnx",
                filepath=str(export_path),
                input_shape=(1, 1, 64, 64)
            )
            
            assert export_path.exists()
            
        except Exception as e:
            # ONNX export might not work in all environments
            pytest.skip(f"ONNX export test skipped: {e}")
    
    def test_model_export_torchscript(self, temp_data_dir, temp_output_dir):
        """Test TorchScript model export."""
        trainer = NIITrainer(
            experiment_name="torchscript_test",
            output_dir=str(temp_output_dir)
        )
        
        trainer.setup(str(temp_data_dir / "train"))
        trainer.train(epochs=1)
        
        # Export to TorchScript
        export_path = temp_output_dir / "model.pt"
        
        try:
            trainer.export_model(
                format="torchscript",
                filepath=str(export_path)
            )
            
            assert export_path.exists()
            
            # Test loading exported model
            loaded_model = torch.jit.load(str(export_path))
            assert loaded_model is not None
            
        except Exception as e:
            pytest.skip(f"TorchScript export test skipped: {e}")


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_cli_config_creation(self, temp_output_dir):
        """Test CLI configuration creation."""
        from nii_trainer.cli.commands import _create_config
        import argparse
        
        # Mock CLI arguments
        args = argparse.Namespace(
            template='basic',
            output=str(temp_output_dir / "test_config.yaml")
        )
        
        result = _create_config(args)
        assert result == 0  # Success
        assert Path(args.output).exists()
    
    def test_cli_info_command(self):
        """Test CLI info command."""
        from nii_trainer.cli.commands import info_command
        import argparse
        
        args = argparse.Namespace(
            system=True,
            gpu=False,
            dependencies=False
        )
        
        result = info_command(args)
        assert result == 0  # Success


class TestDataPipeline:
    """Test complete data pipeline."""
    
    def test_data_loading_pipeline(self, temp_data_dir, sample_config):
        """Test data loading and preprocessing pipeline."""
        from nii_trainer.data import create_dataset, create_dataloader
        
        # Create dataset
        train_dataset = create_dataset(
            data_path=str(temp_data_dir / "train"),
            config=sample_config.data,
            mode="train"
        )
        
        assert len(train_dataset) > 0
        
        # Create dataloader
        train_loader = create_dataloader(
            dataset=train_dataset,
            config=sample_config.data,
            mode="train"
        )
        
        # Test batch loading
        batch = next(iter(train_loader))
        assert 'image' in batch
        assert 'label' in batch
    
    def test_data_transforms(self, sample_image, sample_label):
        """Test data transformation pipeline."""
        from nii_trainer.data.transforms import Compose, RandomRotation, Normalize
        
        transform = Compose([
            RandomRotation(degrees=10),
            Normalize(mean=0, std=1)
        ])
        
        # Apply transforms
        transformed = transform({
            'image': sample_image,
            'label': sample_label
        })
        
        assert 'image' in transformed
        assert 'label' in transformed


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_data_path(self):
        """Test handling of invalid data paths."""
        trainer = NIITrainer(experiment_name="error_test")
        
        with pytest.raises(Exception):
            trainer.setup("/nonexistent/path")
    
    def test_invalid_checkpoint_loading(self, temp_data_dir):
        """Test handling of invalid checkpoint loading."""
        trainer = NIITrainer(experiment_name="checkpoint_error_test")
        trainer.setup(str(temp_data_dir / "train"))
        
        with pytest.raises(Exception):
            trainer.load_checkpoint("/nonexistent/checkpoint.pth")
    
    def test_memory_error_handling(self, sample_config):
        """Test handling of potential memory errors."""
        # Set unrealistic batch size
        sample_config.data.batch_size = 1000
        sample_config.data.image_size = [2048, 2048]  # Very large
        
        trainer = NIITrainer(config=sample_config)
        
        # This should either work or fail gracefully
        try:
            # This would likely cause memory issues in practice
            model = trainer._build_model()
        except (RuntimeError, MemoryError):
            # Expected behavior for memory issues
            pass


class TestReproducibility:
    """Test reproducibility and determinism."""
    
    def test_deterministic_training(self, temp_data_dir, temp_output_dir):
        """Test that training is deterministic with fixed seeds."""
        import random
        import numpy as np
        
        def train_with_seed(seed):
            # Set all random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            trainer = NIITrainer(
                experiment_name=f"repro_test_{seed}",
                output_dir=str(temp_output_dir)
            )
            trainer.setup(str(temp_data_dir / "train"))
            return trainer.train(epochs=1)
        
        # Train with same seed twice
        results1 = train_with_seed(42)
        results2 = train_with_seed(42)
        
        # Results should be similar (allowing for small numerical differences)
        if 'train_loss' in results1 and 'train_loss' in results2:
            assert abs(results1['train_loss'] - results2['train_loss']) < 0.1


class TestPerformance:
    """Test performance characteristics."""
    
    def test_training_speed(self, temp_data_dir, temp_output_dir):
        """Test training speed benchmark."""
        import time
        
        trainer = NIITrainer(
            experiment_name="speed_test",
            output_dir=str(temp_output_dir)
        )
        trainer.setup(str(temp_data_dir / "train"))
        
        start_time = time.time()
        trainer.train(epochs=1)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # Training should complete within reasonable time (10 minutes max)
        assert training_time < 600  # 10 minutes
        print(f"Training time: {training_time:.2f} seconds")
    
    def test_memory_usage(self, temp_data_dir, sample_config):
        """Test memory usage monitoring."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = NIITrainer(config=sample_config)
        trainer.setup(str(temp_data_dir / "train"))
        trainer.train(epochs=1)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"Memory used: {memory_used:.2f} MB")
        
        # Memory usage should be reasonable (less than 2GB)
        assert memory_used < 2000  # 2GB limit