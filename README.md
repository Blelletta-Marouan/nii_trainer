# NII-Trainer: Advanced Medical Image Segmentation Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

NII-Trainer is a comprehensive PyTorch-based framework specifically designed for medical image segmentation using advanced cascaded neural networks. It provides everything you need to train, evaluate, and deploy state-of-the-art segmentation models for medical imaging applications.

## ✨ What Makes NII-Trainer Special?

- 🏥 **Medical Image Native**: Built specifically for medical imaging formats (NIfTI, DICOM, NRRD)
- 🔗 **Cascaded Architecture**: Multi-stage progressive refinement for superior accuracy
- 🚀 **Production Ready**: Complete training pipeline with monitoring, evaluation, and deployment
- 🎯 **Easy to Use**: Simple API for beginners, advanced features for experts
- 🧠 **Smart Training**: Automated mixed precision, early stopping, and hyperparameter optimization
- 📊 **Comprehensive Evaluation**: Multiple metrics, statistical analysis, and visualization tools

## 🚀 Quick Start

### Installation

```bash
pip install nii-trainer
```

### Your First Model in 5 Minutes

```python
import nii_trainer

# Create a quick-start configuration
config = nii_trainer.quick_start_config()

# Load your medical images
train_data = nii_trainer.load_medical_image("path/to/train/images")
val_data = nii_trainer.load_medical_image("path/to/val/images")

# Create and train a cascaded segmentation model
model = nii_trainer.create_cascaded_model(config.model)
trainer = nii_trainer.BaseTrainer(model, config)

# Train the model
results = trainer.fit(train_data, val_data)

# Evaluate performance
evaluator = nii_trainer.create_evaluator(config.evaluation)
metrics = evaluator.evaluate(model, test_data)
print(f"Dice Score: {metrics['dice_score']:.3f}")
```

### Command Line Interface

Train a model with just one command:

```bash
# Quick training with defaults
nii-trainer train --data-dir /path/to/data --output-dir ./results

# Advanced configuration
nii-trainer train --config config.yaml --gpu 0 --epochs 100

# Evaluate trained model
nii-trainer evaluate --model-path ./results/best_model.pth --test-data /path/to/test

# Make predictions on new images
nii-trainer predict --input image.nii.gz --output prediction.nii.gz --model ./results/best_model.pth
```

## 🏗️ Architecture Overview

NII-Trainer implements a sophisticated cascaded approach:

```
Stage 1: Coarse Segmentation
    ├── ResNet/EfficientNet Encoder
    ├── U-Net/FPN Decoder
    └── Initial region localization

Stage 2: Fine Segmentation
    ├── Advanced encoder with attention
    ├── Multi-scale feature fusion
    └── Refined segmentation with Stage 1 guidance

Stage 3: Post-processing (Optional)
    ├── Morphological operations
    ├── Boundary refinement
    └── Quality assurance checks
```

## 📚 Key Components

### 🔧 Core Features

- **Model Architectures**: Cascaded U-Net, Progressive Cascade, Multi-stage architectures
- **Encoders**: ResNet, EfficientNet, DenseNet, with pretrained weights
- **Decoders**: U-Net, FPN, DeepLab variants
- **Loss Functions**: Dice, Focal, Tversky, IoU, Boundary, and composite losses
- **Data Processing**: Automatic preprocessing, augmentation, and normalization

### 📊 Training & Evaluation

- **Smart Training**: Mixed precision, distributed training, gradient clipping
- **Advanced Optimizers**: Adam, AdamW, SGD with custom schedulers
- **Comprehensive Metrics**: Dice, IoU, Hausdorff distance, surface distance
- **Visualization**: Training curves, prediction overlays, attention maps

### 🔌 Extensibility

- **Plugin Architecture**: Easy to add custom models, losses, and metrics
- **Configuration System**: YAML-based configuration with validation
- **Registry System**: Automatic component discovery and registration

## 🎯 Supported Use Cases

| Application | Description | Key Features |
|-------------|-------------|--------------|
| **Organ Segmentation** | Liver, kidney, heart, brain | Multi-organ support, anatomical priors |
| **Tumor Detection** | Cancer detection and delineation | Boundary refinement, uncertainty estimation |
| **Pathology Analysis** | Tissue classification | Multi-class segmentation, attention mechanisms |
| **Research** | Custom medical AI applications | Flexible architecture, extensive customization |

## 📖 Documentation & Examples

### 📝 Detailed Guides

- [📚 User Guide](docs/user_guide.md) - Complete usage tutorial
- [⚙️ Configuration Guide](docs/configuration.md) - All configuration options
- [🔧 API Reference](docs/api/) - Complete API documentation
- [🚀 Advanced Features](docs/advanced/) - Expert-level features

### 💡 Example Projects

```python
# Example 1: Liver Segmentation
from nii_trainer import create_model, BaseTrainer

config = {
    'model_name': 'cascaded_segmentation',
    'num_stages': 2,
    'input_channels': 1,
    'num_classes': 2
}

model = create_model(config)
trainer = BaseTrainer(model)
trainer.fit(train_loader, val_loader)

# Example 2: Multi-organ Segmentation
config = {
    'model_name': 'progressive_cascade',
    'num_stages': 3,
    'num_classes': 5,  # Background + 4 organs
    'use_attention': True,
    'use_deep_supervision': True
}

# Example 3: Custom Loss Function
from nii_trainer.training.losses import DiceFocalLoss

custom_loss = DiceFocalLoss(
    dice_weight=0.7,
    focal_weight=0.3,
    focal_gamma=2.0
)
```
## ⚡ Advanced Features

### 🎛️ Hyperparameter Optimization

```python
from nii_trainer.experimental import AutoTuner

# Automatic hyperparameter tuning
tuner = AutoTuner(
    model_config=config,
    search_space='default',
    n_trials=50
)

best_params = tuner.optimize(train_data, val_data)
```

### 📊 Experiment Tracking

```python
# Integration with popular tracking tools
trainer = BaseTrainer(
    model=model,
    logger='wandb',  # or 'tensorboard', 'mlflow'
    experiment_name='liver_segmentation_v2'
)
```

### 🔄 Model Ensemble

```python
# Combine multiple models for better performance
from nii_trainer.experimental import ModelEnsemble

ensemble = ModelEnsemble([model1, model2, model3])
predictions = ensemble.predict(test_images)
```

## 🛠️ Installation Options

### Standard Installation
```bash
pip install nii-trainer
```

### Development Installation
```bash
git clone https://github.com/your-org/nii-trainer.git
cd nii-trainer
pip install -e ".[dev]"
```

### Docker Installation
```bash
docker pull nii-trainer/nii-trainer:latest
docker run -it --gpus all nii-trainer/nii-trainer:latest
```

## 🤝 Community & Support

- 💬 **Discord**: [Join our community](https://discord.gg/nii-trainer)
- 📧 **Email**: support@nii-trainer.org
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/nii-trainer/issues)
- 📖 **Documentation**: [Full docs](https://nii-trainer.readthedocs.io)

## 🚧 Roadmap

- [ ] **Q2 2024**: 3D segmentation support
- [ ] **Q3 2024**: Federated learning capabilities
- [ ] **Q4 2024**: Real-time inference optimization
- [ ] **Q1 2025**: Mobile deployment support

## 📄 Citation

If you use NII-Trainer in your research, please cite:

```bibtex
@software{nii_trainer,
  title={NII-Trainer: Advanced Medical Image Segmentation Framework},
  author={NII-Trainer Development Team},
  year={2024},
  url={https://github.com/your-org/nii-trainer}
}
```

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Medical imaging community for datasets and feedback
- All contributors and users who help improve NII-Trainer

---

**Ready to get started?** Check out our [Quick Start Guide](docs/quickstart.md) or try the [Interactive Tutorial](https://colab.research.google.com/github/your-org/nii-trainer/blob/main/examples/tutorial.ipynb)!