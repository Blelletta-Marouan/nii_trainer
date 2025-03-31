# NII Trainer: Advanced Cascaded Medical Image Segmentation

A highly flexible framework for training cascaded neural networks on medical imaging data, with a focus on hierarchical multi-class segmentation. The framework features a novel binary-first approach where the initial stage performs foreground/background separation before subsequent stages handle fine-grained class segmentation.

## 🌟 Key Features

- **Binary-First Cascade Architecture**: Initial stage performs binary segmentation, followed by fine-grained class segmentation
- **Flexible Backbone Support**: Multiple encoder options (MobileNetV2, ResNet18/50, EfficientNet)
- **Configurable Stage Architecture**: Each stage can have different:
  - Encoder backbones
  - Number of layers
  - Skip connection configurations
- **Advanced Training Features**:
  - Mixed precision training
  - Reward-based loss functions
  - Automatic class balancing
  - Comprehensive metrics tracking

## 📦 Installation

```bash
git clone https://github.com/Blelletta-Marouan/nii_trainer
cd nii_trainer
pip install -e .
```

## 🚀 Quick Start

```python
from nii_trainer.configs import create_liver_config
from nii_trainer.models import FlexibleCascadedUNet
from nii_trainer.data import NiiPreprocessor

# 1. Create configuration
config = create_liver_config()

# 2. Initialize model and trainer
model = FlexibleCascadedUNet(config.cascade)
trainer = ModelTrainer(config, model, ...)

# 3. Train
trainer.train()
```

## 📋 Step-by-Step Guide

### 1. Data Preparation

The framework expects your data in NIfTI format. Here are three ways to prepare your data:

```python
# Example 1: Basic preprocessing
preprocessor = NiiPreprocessor(config.data)
preprocessor.extract_slices(
    input_dir="raw_data",
    output_dir="processed_data",
    window_width=180,
    window_level=50
)

# Example 2: With class balancing
preprocessor.extract_slices(
    input_dir="raw_data",
    output_dir="processed_data",
    balance_classes=True,
    class_ratios={'background': 1, 'liver': 2, 'tumor': 3}
)

# Example 3: With data augmentation
preprocessor.extract_slices(
    input_dir="raw_data",
    output_dir="processed_data",
    augment=True,
    aug_params={
        'rotation_range': 15,
        'zoom_range': 0.1,
        'elastic_deformation': True
    }
)
```

### 2. Configuration Setup

Three ways to configure your cascade:

```python
# Example 1: Liver-tumor segmentation
config = create_liver_config()

# Example 2: Brain tumor segmentation
config = TrainerConfig(
    data=DataConfig(
        classes=["background", "white_matter", "tumor", "edema"],
        img_size=(256, 256)
    ),
    cascade=CascadeConfig(stages=[
        StageConfig(
            input_classes=["background", "foreground"],
            target_class="foreground",
            is_binary=True
        ),
        StageConfig(
            input_classes=["foreground"],
            target_class="white_matter"
        ),
        StageConfig(
            input_classes=["white_matter"],
            target_class="tumor"
        )
    ])
)

# Example 3: Custom multi-organ segmentation
config = TrainerConfig(
    data=DataConfig(...),
    cascade=CascadeConfig(
        stages=[
            StageConfig(
                input_classes=["background", "foreground"],
                target_class="foreground",
                encoder_type="efficientnet",
                is_binary=True
            ),
            # Add more stages as needed
        ]
    )
)
```

### 3. Model Training

Three training approaches:

```python
# Example 1: Basic training
trainer = ModelTrainer(config, model)
trainer.train()

# Example 2: With continued training
trainer.load_checkpoint('previous_model.pth')
trainer.train(
    additional_epochs=20,
    fine_tune=True
)

# Example 3: With curriculum learning
trainer.train_with_curriculum(
    stage_schedule=[
        (0, 10),   # Train first stage for 10 epochs
        (1, 15),   # Add second stage for 15 epochs
        (2, 20)    # Add final stage for 20 epochs
    ]
)
```

## 🔧 Advanced Usage

### Custom Stage Configuration

Configure each stage independently:

```python
stage_config = StageConfig(
    input_classes=["background", "foreground"],
    target_class="foreground",
    encoder_type="mobilenet_v2",
    encoder_layers=5,
    decoder_layers=5,
    is_binary=True,
    threshold=0.5
)
```

### Reward-based Training

```python
loss_config = LossConfig(
    weight_bce=0.5,
    weight_dice=0.5,
    reward_coef=0.1,
    stage_weights=[1.0, 1.2, 1.5]
)
```

## 📊 Visualization

```python
from nii_trainer.visualization import SegmentationVisualizer

visualizer = SegmentationVisualizer(config.data.classes)
visualizer.plot_predictions(predictions, targets)
visualizer.plot_metrics(metrics_history)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{nii_trainer2025,
    title={NII Trainer: Advanced Cascaded Medical Image Segmentation},
    author={BLELLETTA Marouan},
    year={2025},
    url={https://github.com/Blelletta-Marouan/nii_trainer}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.