"""
Test fixtures for NII-Trainer test suite.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import SimpleITK as sitk

from nii_trainer.core.config import GlobalConfig, create_default_config
from nii_trainer.models import create_model


@pytest.fixture(scope="session")
def sample_config():
    """Create a sample configuration for testing."""
    config = create_default_config()
    config.model.model_name = "test_model"
    config.model.num_stages = 2
    config.training.max_epochs = 2  # Quick training for tests
    config.training.batch_size = 2
    config.data.batch_size = 2
    config.data.image_size = [64, 64]  # Small size for testing
    return config


@pytest.fixture(scope="session")
def sample_image():
    """Create a sample medical image for testing."""
    # Create a 3D image with realistic medical image properties
    image_array = np.random.randint(-1000, 1000, (64, 64, 32), dtype=np.int16)
    
    # Add some structure (simulated organ)
    center = (32, 32, 16)
    for z in range(32):
        for y in range(64):
            for x in range(64):
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if distance < 15:  # Organ region
                    image_array[y, x, z] = np.random.randint(50, 200)
    
    # Convert to SimpleITK image
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 2.0))  # Realistic spacing
    image.SetOrigin((0.0, 0.0, 0.0))
    
    return image


@pytest.fixture(scope="session")
def sample_label():
    """Create a sample segmentation label for testing."""
    # Create binary segmentation mask
    label_array = np.zeros((64, 64, 32), dtype=np.uint8)
    
    # Add segmentation region
    center = (32, 32, 16)
    for z in range(32):
        for y in range(64):
            for x in range(64):
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if distance < 12:  # Slightly smaller than organ
                    label_array[y, x, z] = 1
    
    # Convert to SimpleITK image
    label = sitk.GetImageFromArray(label_array)
    label.SetSpacing((1.0, 1.0, 2.0))
    label.SetOrigin((0.0, 0.0, 0.0))
    
    return label


@pytest.fixture
def temp_data_dir(sample_image, sample_label):
    """Create temporary data directory with sample data."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (temp_dir / split / 'images').mkdir(parents=True)
            (temp_dir / split / 'labels').mkdir(parents=True)
            
            # Create sample files for each split
            for i in range(3):  # 3 samples per split
                image_path = temp_dir / split / 'images' / f'case_{i:03d}.nii.gz'
                label_path = temp_dir / split / 'labels' / f'case_{i:03d}.nii.gz'
                
                sitk.WriteImage(sample_image, str(image_path))
                sitk.WriteImage(sample_label, str(label_path))
        
        yield temp_dir
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model(sample_config):
    """Create a mock model for testing."""
    return create_model(sample_config.model)


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_tensor():
    """Create sample tensor for testing."""
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def sample_3d_tensor():
    """Create sample 3D tensor for testing."""
    return torch.randn(2, 1, 32, 64, 64)


@pytest.fixture
def sample_batch():
    """Create sample training batch."""
    return {
        'image': torch.randn(2, 1, 64, 64),
        'label': torch.randint(0, 2, (2, 1, 64, 64)).float(),
        'case_id': ['case_001', 'case_002']
    }


@pytest.fixture(scope="session")
def test_checkpoint_path():
    """Create a temporary checkpoint file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
    
    # Create dummy checkpoint data
    checkpoint = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'epoch': 10,
        'loss': 0.5,
        'metrics': {'dice': 0.85, 'iou': 0.75}
    }
    
    torch.save(checkpoint, temp_file.name)
    
    yield temp_file.name
    
    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture
def mock_dataloader(sample_batch):
    """Create mock dataloader for testing."""
    class MockDataLoader:
        def __init__(self, batch, num_batches=3):
            self.batch = batch
            self.num_batches = num_batches
            
        def __iter__(self):
            for _ in range(self.num_batches):
                yield self.batch
                
        def __len__(self):
            return self.num_batches
    
    return MockDataLoader(sample_batch)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_config_file(sample_config):
    """Create temporary configuration file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    sample_config.to_yaml(temp_file.name)
    
    yield temp_file.name
    
    Path(temp_file.name).unlink(missing_ok=True)


class MockTrainer:
    """Mock trainer class for testing."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_metrics = {}
        
    def train(self):
        return {'train_loss': 0.5, 'val_dice': 0.85}
        
    def evaluate(self):
        return {'dice': 0.85, 'iou': 0.75, 'hausdorff': 2.5}


@pytest.fixture
def mock_trainer(sample_config):
    """Create mock trainer for testing."""
    return MockTrainer(sample_config)


class MockEvaluator:
    """Mock evaluator class for testing."""
    
    def __init__(self, config):
        self.config = config
        
    def evaluate(self, model, data_loader=None):
        return {
            'dice': 0.85,
            'iou': 0.75,
            'hausdorff': 2.5,
            'precision': 0.88,
            'recall': 0.82
        }


@pytest.fixture
def mock_evaluator(sample_config):
    """Create mock evaluator for testing."""
    return MockEvaluator(sample_config.evaluation)