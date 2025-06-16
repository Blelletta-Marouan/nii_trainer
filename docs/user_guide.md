# NII-Trainer User Guide

This comprehensive guide will help you get started with NII-Trainer and explore its advanced features for medical image segmentation.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Training Models](#training-models)
6. [Evaluation and Testing](#evaluation-and-testing)
7. [Model Deployment](#model-deployment)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)

## Installation

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Basic Installation

```bash
pip install nii-trainer
```

### Development Installation

For contributors and advanced users:

```bash
git clone https://github.com/your-org/nii-trainer.git
cd nii-trainer
pip install -e ".[dev]"
```

### Verify Installation

```bash
nii-trainer info
```

This should display your system information and confirm the installation.

## Basic Usage

### Simple Training Example

```python
from nii_trainer import NIITrainer

# Initialize trainer
trainer = NIITrainer(
    experiment_name="my_segmentation",
    output_dir="./experiments"
)

# Setup data
trainer.setup(
    train_data_path="/path/to/training/data",
    val_data_path="/path/to/validation/data"
)

# Train the model
results = trainer.train(epochs=100)

# Evaluate on test data
test_results = trainer.evaluate(
    test_data_path="/path/to/test/data",
    checkpoint_path="./experiments/best_model.pth"
)
```

### Quick Training Function

For rapid prototyping:

```python
from nii_trainer import quick_train

trainer = quick_train(
    train_data_path="/path/to/train",
    val_data_path="/path/to/val",
    epochs=50,
    experiment_name="quick_experiment"
)
```

## Configuration

### Using Configuration Files

Create a configuration file to customize your training:

```bash
nii-trainer config create --template advanced --output my_config.yaml
```

Example configuration:

```yaml
model:
  model_name: "cascaded_unet"
  num_stages: 2
  encoder: "resnet50"
  decoder: "unet"
  use_attention: true
  
training:
  max_epochs: 100
  learning_rate: 0.001
  batch_size: 4
  optimizer: "adam"
  scheduler: "cosine"
  mixed_precision: true
  
data:
  image_size: [512, 512]
  augmentation: true
  normalize: true
  
evaluation:
  metrics: ["dice", "iou", "hausdorff"]
  save_predictions: true
```

Use the configuration:

```python
trainer = NIITrainer(config="my_config.yaml")
```

### Configuration Templates

Three templates are available:

- **basic**: Default settings for beginners
- **advanced**: Optimized settings with attention and mixed precision
- **research**: Full feature set for research applications

## Data Preparation

### Data Structure

Organize your data in the following structure:

```
data/
├── train/
│   ├── images/
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   └── labels/
│       ├── case_001.nii.gz
│       ├── case_002.nii.gz
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Supported Formats

- **NIfTI**: `.nii`, `.nii.gz` (preferred for medical imaging)
- **DICOM**: `.dcm` files
- **NRRD**: `.nrrd` files
- **Other**: Any format supported by SimpleITK

### Data Preprocessing

NII-Trainer automatically handles:

- **Intensity normalization**: HU value windowing for CT scans
- **Spatial resampling**: Consistent voxel spacing
- **Image registration**: Alignment of images and labels
- **Augmentation**: Rotation, translation, elastic deformation

Custom preprocessing:

```python
from nii_trainer.data import VolumeProcessor

processor = VolumeProcessor(
    target_spacing=(1.0, 1.0, 2.0),
    target_size=(512, 512, 64),
    intensity_range=(-1000, 1000)
)

processed_image = processor.process(image_path)
```

## Training Models

### Basic Training

```python
from nii_trainer import NIITrainer

trainer = NIITrainer(experiment_name="liver_segmentation")
trainer.setup("/path/to/train", "/path/to/val")

# Start training
results = trainer.train(epochs=100)

# Training progress is automatically logged
```

### Advanced Training

```python
from nii_trainer.api import AdvancedTrainer
from nii_trainer.core.config import create_default_config

config = create_default_config()
config.training.mixed_precision = True
config.training.distributed = True

trainer = AdvancedTrainer(config)
trainer.build_model()
trainer.setup_training()

# Add custom callbacks
def lr_callback(phase, trainer):
    if phase == "on_epoch_end":
        print(f"Learning rate: {trainer.get_lr()}")

trainer.add_callback(lr_callback)

# Train with validation
results = trainer.train_with_validation(train_loader, val_loader)
```

### Training Strategies

1. **Standard Training**: Single-stage training
2. **Progressive Training**: Start with coarse, add fine stages
3. **Transfer Learning**: Pre-trained encoders
4. **Multi-GPU Training**: Distributed training support

### Monitoring Training

Training progress is automatically logged to:

- **Console**: Real-time metrics
- **TensorBoard**: Visualization and graphs
- **Files**: CSV logs and checkpoints

Access TensorBoard:

```bash
tensorboard --logdir ./experiments/logs
```

### Early Stopping

Configure early stopping to prevent overfitting:

```python
config.training.early_stopping_patience = 10
config.training.early_stopping_metric = "val_dice"
config.training.early_stopping_mode = "max"
```

## Evaluation and Testing

### Basic Evaluation

```python
# Evaluate with saved model
results = trainer.evaluate(
    test_data_path="/path/to/test",
    checkpoint_path="best_model.pth"
)

print(f"Dice Score: {results['dice']:.4f}")
print(f"IoU Score: {results['iou']:.4f}")
```

### Comprehensive Evaluation

```python
from nii_trainer.evaluation import SegmentationEvaluator

evaluator = SegmentationEvaluator(config.evaluation)

# Evaluate with multiple metrics
results = evaluator.evaluate_comprehensive(
    model=model,
    test_loader=test_loader,
    compute_hausdorff=True,
    bootstrap_samples=1000
)

# Generate evaluation report
evaluator.generate_report(results, "evaluation_report.html")
```

### Cross-Validation

```python
from nii_trainer.evaluation import cross_validate

cv_results = cross_validate(
    data_path="/path/to/data",
    config=config,
    n_folds=5,
    stratify=True
)
```

### Statistical Analysis

```python
# Compare multiple models
from nii_trainer.evaluation import statistical_comparison

comparison = statistical_comparison(
    results_a=results_model_a,
    results_b=results_model_b,
    metric="dice",
    test="wilcoxon"
)
```

## Model Deployment

### Export Models

Export trained models for deployment:

```python
# Export to ONNX
trainer.export_model(
    format="onnx",
    filepath="model.onnx",
    input_shape=(1, 1, 512, 512)
)

# Export to TorchScript
trainer.export_model(
    format="torchscript",
    filepath="model.pt"
)
```

### Command Line Export

```bash
nii-trainer export \
    --checkpoint best_model.pth \
    --output model.onnx \
    --format onnx \
    --input-shape 1 1 512 512
```

### Inference

```python
# Load exported model for inference
import torch

model = torch.jit.load("model.pt")
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
```

### Batch Prediction

```bash
# Predict on multiple images
nii-trainer predict \
    --input /path/to/images/ \
    --output /path/to/predictions/ \
    --checkpoint model.pth \
    --batch-size 4
```

## Advanced Features

### AutoML

Automatically find the best model configuration:

```python
from nii_trainer.api.experimental import AutoML

automl = AutoML(time_budget=3600)  # 1 hour budget

results = automl.auto_train(
    train_data_path="/path/to/train",
    val_data_path="/path/to/val"
)

print(f"Best configuration: {results['best_config']}")
print(f"Best score: {results['best_score']}")
```

### Neural Architecture Search

```python
from nii_trainer.api.experimental import NeuralArchitectureSearch

nas = NeuralArchitectureSearch(search_strategy="evolutionary")

search_space = {
    'encoder': ['resnet18', 'resnet50', 'efficientnet-b0'],
    'decoder': ['unet', 'fpn'],
    'num_stages': [1, 2, 3],
    'use_attention': [True, False]
}

results = nas.search_architecture(
    search_space=search_space,
    n_architectures=50,
    train_data_path="/path/to/train",
    val_data_path="/path/to/val"
)
```

### Hyperparameter Tuning

```python
from nii_trainer.api.advanced import HyperparameterTuner

tuner = HyperparameterTuner(
    objective_metric="val_dice",
    direction="maximize",
    n_trials=100
)

search_space = {
    'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2},
    'batch_size': {'type': 'categorical', 'choices': [2, 4, 8, 16]},
    'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'adamw']}
}

tuner.define_search_space(search_space)
best_params = tuner.optimize()
```

### Federated Learning

```python
from nii_trainer.api.experimental import FederatedTrainer

fed_trainer = FederatedTrainer(aggregation_strategy="fedavg")

# Initialize global model
fed_trainer.initialize_global_model(model_config)

# Add clients
fed_trainer.add_client("hospital_1", "/data/hospital1", data_size=1000)
fed_trainer.add_client("hospital_2", "/data/hospital2", data_size=800)

# Run federated training
results = fed_trainer.federated_train(
    n_rounds=10,
    clients_per_round=2,
    local_epochs=5
)
```

### Custom Components

Create custom encoders:

```python
from nii_trainer.models.encoders import BaseEncoder

class MyCustomEncoder(BaseEncoder):
    def __init__(self, in_channels=1, features=[64, 128, 256]):
        super().__init__()
        # Implement your encoder
        
    def forward(self, x):
        # Forward pass implementation
        return encoded_features

# Register custom component
from nii_trainer.core.registry import ENCODER_REGISTRY
ENCODER_REGISTRY.register("my_encoder", MyCustomEncoder)
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```

Solutions:
- Reduce batch size: `config.data.batch_size = 2`
- Enable gradient checkpointing: `config.training.gradient_checkpointing = True`
- Use mixed precision: `config.training.mixed_precision = True`

**2. Data Loading Errors**
```
FileNotFoundError: Image file not found
```

Solutions:
- Check file paths are correct
- Verify file permissions
- Ensure data structure matches expected format

**3. Model Architecture Errors**
```
RuntimeError: size mismatch
```

Solutions:
- Check input image dimensions
- Verify model configuration
- Ensure encoder-decoder compatibility

### Performance Optimization

**Speed up training:**
- Use mixed precision training
- Increase number of data loading workers
- Use appropriate batch size for your GPU
- Enable benchmark mode for consistent input sizes

**Reduce memory usage:**
- Lower batch size
- Use gradient checkpointing
- Optimize data loading pipeline
- Clear GPU cache periodically

### Getting Help

1. **Check logs**: Training logs contain detailed error information
2. **Documentation**: Refer to API documentation for detailed parameters
3. **GitHub Issues**: Report bugs or ask questions
4. **Community**: Join discussions for tips and best practices

### Debug Mode

Enable verbose logging for detailed debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via CLI
nii-trainer train --verbose ...
```

Use debugging utilities:

```python
from nii_trainer.utils.debugging import debug_tensor, check_gradients

# Debug tensor values
debug_tensor(model_output, "model_output")

# Check gradient flow
grad_info = check_gradients(model)
print(f"Gradient issues: {grad_info['nan_gradients']}")
```