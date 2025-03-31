# NII Trainer: Advanced Cascaded Medical Image Segmentation

A powerful and flexible framework for training cascaded neural networks on medical imaging data (NIfTI format). This framework implements a novel binary-first approach for hierarchical multi-class segmentation, where the initial stage performs foreground/background separation before subsequent stages handle fine-grained class segmentation.

## 🌟 Key Features

- **Binary-First Cascade Architecture**: 
  - Initial stage performs binary segmentation
  - Subsequent stages handle fine-grained class segmentation
  - Hierarchical learning approach

- **Flexible Model Architecture**:
  - Multiple encoder backbones (MobileNetV2, ResNet18/50, EfficientNet)
  - Configurable number of encoder/decoder layers per stage
  - Optional skip connections
  - Independent stage configurations

- **Advanced Training Features**:
  - Mixed precision training for faster execution
  - Reward-based loss functions
  - Automatic class balancing
  - Comprehensive metrics tracking
  - Curriculum learning support
  - Checkpoint management

- **Visualization and Monitoring**:
  - Real-time training metrics
  - Confusion matrices
  - Segmentation overlays
  - Per-class performance metrics

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Blelletta-Marouan/nii_trainer
cd nii_trainer

# Install in development mode
pip install -e .

# Install required dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

Here's a minimal example to get started:

```python
from nii_trainer.configs import create_liver_config, TrainerConfig
from nii_trainer.models import FlexibleCascadedUNet, ModelTrainer
from nii_trainer.data import NiiPreprocessor, MultiClassSegDataset
from torch.utils.data import DataLoader

# 1. Create basic configuration
config = create_liver_config()

# 2. Preprocess data
preprocessor = NiiPreprocessor(config.data)
train_pairs = preprocessor.extract_slices(
    volume_path="path/to/volume.nii",
    segmentation_path="path/to/segmentation.nii",
    output_dir="processed_data"
)

# 3. Create datasets and dataloaders
train_dataset = MultiClassSegDataset(
    data_dir="processed_data/train",
    class_map=config.data.class_map
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 4. Initialize model and trainer
model = FlexibleCascadedUNet(config.cascade)
trainer = ModelTrainer(config, model, train_loader, val_loader)

# 5. Train
trainer.train()
```

## 📋 Comprehensive Guide

### 1. Data Preparation

The framework supports multiple data preparation approaches:

```python
# Basic preprocessing with windowing (for CT scans)
preprocessor = NiiPreprocessor(
    DataConfig(
        window_params={"width": 180, "level": 50},
        skip_empty=True,  # Skip slices without annotations
        slice_step=2  # Take every 2nd slice
    )
)

# Advanced preprocessing with class balancing
preprocessor = NiiPreprocessor(
    DataConfig(
        img_size=(512, 512),
        balance_dataset=True,
        class_map={
            "background": 0,
            "liver": 1,
            "tumor": 2,
            "vessel": 3
        },
        augmentation_params={
            "rotation_range": (-30, 30),
            "zoom_range": (0.9, 1.1),
            "brightness_range": (0.8, 1.2),
            "elastic_deformation": True
        }
    )
)

# Process multiple volumes with custom slice selection
for volume, segmentation in data_pairs:
    preprocessor.extract_slices(
        volume_path=volume,
        segmentation_path=segmentation,
        output_dir="processed_data",
        slice_indices=range(20, 80, 2)  # Custom slice range
    )
```

### 2. Model Configuration

Examples of different model configurations:

```python
# Basic single-organ segmentation
basic_config = TrainerConfig(
    data=DataConfig(
        classes=["background", "liver"],
        img_size=(256, 256)
    ),
    cascade=CascadeConfig(
        stages=[
            StageConfig(
                input_classes=["background", "liver"],
                target_class="liver",
                encoder_type="mobilenet_v2",
                is_binary=True
            )
        ],
        in_channels=1,
        initial_features=32
    )
)

# Advanced multi-organ segmentation
advanced_config = TrainerConfig(
    data=DataConfig(
        classes=["background", "liver", "tumor", "vessel"],
        img_size=(512, 512),
        window_params={"width": 180, "level": 50}
    ),
    cascade=CascadeConfig(
        stages=[
            # Binary segmentation stage
            StageConfig(
                input_classes=["background", "liver"],
                target_class="liver",
                encoder_type="efficientnet",
                encoder_layers=5,
                decoder_layers=5,
                skip_connections=True,
                dropout_rate=0.3,
                is_binary=True
            ),
            # Liver segmentation stage
            StageConfig(
                input_classes=["liver"],
                target_class="tumor",
                encoder_type="resnet50",
                encoder_layers=4,
                decoder_layers=4
            ),
            # Tumor segmentation stage
            StageConfig(
                input_classes=["liver"],
                target_class="vessel",
                encoder_type="mobilenet_v2",
                encoder_layers=3,
                decoder_layers=3
            )
        ],
        in_channels=1,
        initial_features=64,
        feature_growth=2.0,
        pretrained=True
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        epochs=100,
        patience=10,
        mixed_precision=True,
        batch_accumulation=2
    ),
    loss=LossConfig(
        weight_bce=0.5,
        weight_dice=0.5,
        focal_gamma=2.0,
        reward_coef=0.1,
        stage_weights=[1.0, 1.2, 1.5],
        class_weights={
            "background": 1.0,
            "liver": 2.0,
            "tumor": 4.0,
            "vessel": 3.0
        }
    )
)
```

### 3. Advanced Training Features

#### Curriculum Learning

```python
trainer.train_with_curriculum(
    stage_schedule=[
        (0, 10),   # Train binary stage for 10 epochs
        (1, 15),   # Add liver stage for 15 epochs
        (2, 20)    # Add tumor stage for 20 epochs
    ],
    learning_rates=[1e-3, 5e-4, 1e-4],
    stage_freezing=[False, True, True]  # Freeze previous stages
)
```

#### Checkpointing and Transfer Learning

```python
# Save checkpoints during training
trainer.save_checkpoint(is_best=True)

# Resume training from checkpoint
trainer.load_checkpoint('path/to/checkpoint.pth')
trainer.train(
    additional_epochs=20,
    fine_tune=True,
    learning_rate=1e-5
)
```

#### Custom Training Monitoring

```python
from nii_trainer.visualization import SegmentationVisualizer

visualizer = SegmentationVisualizer(
    class_names=config.data.classes,
    save_dir="visualizations"
)

# Monitor training progress
visualizer.visualize_batch(
    images, predictions, targets,
    max_samples=4,
    save_dir="training_progress"
)

# Plot detailed metrics
visualizer.plot_metrics({
    'Train Loss': trainer.metrics_history['train_loss'],
    'Val Loss': trainer.metrics_history['val_loss'],
    'Dice Scores': trainer.metrics_history['dice_scores']
})

# Generate confusion matrix
visualizer.plot_confusion_matrix(
    predictions=model.get_predictions(outputs),
    targets=targets,
    save_path="confusion_matrix.png"
)
```

### 4. Metrics and Evaluation

The framework provides comprehensive evaluation metrics:

```python
from nii_trainer.utils.metrics import calculate_class_metrics

# Calculate per-class metrics
metrics_df = calculate_class_metrics(
    predictions=model_outputs,
    targets=ground_truth,
    class_names=config.data.classes
)

# Calculate volumetric metrics
vol_metrics = calculate_volumetric_metrics(
    pred_slices=predictions,
    target_slices=targets,
    spacing=(1.5, 1.5, 2.0)  # Physical spacing in mm
)

print("Per-class metrics:")
print(metrics_df)
print("\nVolumetric metrics:")
print(vol_metrics)
```

## 🤝 Contributing

Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to development.

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{nii_trainer2025,
    title={NII Trainer: Advanced Cascaded Medical Image Segmentation},
    author={BLELLETTA Marouan},
    year={2025},
    url={https://github.com/Blelletta-Marouan/nii_trainer},
    version={0.1.0},
    description={A flexible framework for training cascaded neural networks on medical imaging data}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.