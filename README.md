# NII Trainer: Advanced Cascaded Medical Image Segmentation

A powerful and flexible framework for training cascaded neural networks on medical imaging data (NIfTI format). This framework implements a novel binary-first approach for hierarchical multi-class segmentation, where the initial stage performs foreground/background separation before subsequent stages handle fine-grained class segmentation.

![NII Trainer Overview](https://via.placeholder.com/800x400?text=NII+Trainer+Overview)

## 🌟 Key Features

- **Binary-First Cascade Architecture**: 
  - Initial stage performs binary segmentation
  - Subsequent stages handle fine-grained class segmentation
  - Hierarchical learning approach for improved performance
  - Flexible multi-stage processing pipeline

- **Flexible Model Architecture**:
  - Multiple encoder backbones (MobileNetV2, ResNet18/50, EfficientNet)
  - Configurable number of encoder/decoder layers per stage
  - Optional skip connections and attention mechanisms
  - Independent stage configurations for targeted optimization
  - Support for 2D and 3D models

- **Advanced Training Features**:
  - Mixed precision training for faster execution
  - Reward-based loss functions with adaptive weighting
  - Automatic class balancing for imbalanced datasets
  - Comprehensive metrics tracking and visualization
  - Curriculum learning support with configurable stage progression
  - Checkpoint management and experiment tracking

- **Visualization and Monitoring**:
  - Real-time training metrics and performance dashboards
  - Interactive confusion matrices and error analysis
  - Segmentation overlays with multi-class visualization
  - Per-class performance metrics and volumetric analysis
  - Uncertainty visualization for model confidence assessment

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Blelletta-Marouan/nii-trainer
cd nii-trainer

# Install in development mode
pip install -e .

# Install required dependencies
pip install -r requirements.txt

# Optional: Install visualization dependencies
pip install -r visualization_requirements.txt
```

## 🚀 Quick Start

Here's a minimal example to get started with NII Trainer:

```python
from nii_trainer.configs import create_liver_config
from nii_trainer.utils import Experiment

# Create default configuration for liver segmentation
config = create_liver_config()

# Create and run experiment
experiment = Experiment(
    config=config,
    experiment_name="liver_segmentation_demo",
    base_dir="experiments"
)

# Run the full pipeline (data processing, training, evaluation)
results = experiment.run(
    volume_dir="path/to/volumes",
    segmentation_dir="path/to/segmentations",
    process_data=True,
    curriculum=True,  # Use curriculum learning
    force_overwrite=False
)

# Print results and visualize
experiment.print_metrics()
experiment.visualize()
```

## 📋 Comprehensive Guide

### 1. Data Processing Module

NII Trainer offers flexible tools for processing NIfTI medical imaging data with extensive configuration options.

#### Basic Data Processing

The most straightforward way to process your NIfTI data:

```python
from nii_trainer.data import NiiPreprocessor
from nii_trainer.configs.config import DataConfig

# Configure basic preprocessing parameters
data_config = DataConfig(
    base_dir="raw_data",
    output_dir="processed_data",
    classes=["background", "liver", "tumor"],
    class_map={"background": 0, "liver": 1, "tumor": 2},
    
    # Core preprocessing parameters
    img_size=(512, 512),               # Target image size
    slice_step=1,                      # Process every slice (use 2+ to skip slices)
    skip_empty=True,                   # Skip slices without annotations
    train_val_test_split=(0.7, 0.15, 0.15),  # Dataset split ratios
    
    # CT windowing parameters (for optimal tissue visualization)
    window_params={"window_width": 180, "window_level": 50}
)

# Initialize preprocessor and process a volume
preprocessor = NiiPreprocessor(data_config)
preprocessor.extract_slices(
    volume_path="volumes/volume-0.nii",
    segmentation_path="segmentations/segmentation-0.nii",
    output_dir="processed_data"
)
```

#### Advanced Data Processing

For more complex processing needs, use the `BatchProcessor` for greater control:

```python
from nii_trainer.data import BatchProcessor

# Create batch processor with customizable parameters
processor = BatchProcessor(
    img_size=(512, 512),
    window_params={"window_width": 180, "window_level": 50},
    skip_empty=True,
    slice_step=2,  # Process every 2nd slice
    train_val_test_split=(0.6, 0.2, 0.2),
    
    # Customize file matching patterns
    segmentation_pattern="segmentation-{}.nii",  # Uses {} as placeholder for ID
    volume_pattern="volume-{}.nii"
)

# Process a batch of volumes
processor.process_batch(
    volume_dir="volumes",
    segmentation_dir="segmentations",
    output_dir="processed_data",
    file_pattern="*.nii",
    max_volumes=10,  # Limit the number of volumes to process
    skip_existing=True,  # Skip already processed volumes
    include_val=True,
    include_test=True,
    force_overwrite=False
)

# Or process with custom patterns
processor.process_with_naming_convention(
    base_dir="datasets/liver_tumor",
    segmentation_dir="datasets/liver_tumor/labels",
    output_dir="processed_data",
    naming_convention={
        "volume": "patient{}_ct.nii",
        "segmentation": "patient{}_seg.nii"
    }
)
```

#### Tissue-Specific Parameters

Configure different windowing parameters for different tissue types:

```python
from nii_trainer.data import NiiPreprocessor
from nii_trainer.configs.config import DataConfig

# Advanced preprocessing with tissue-specific windowing
data_config = DataConfig(
    base_dir="raw_data",
    output_dir="processed_data",
    classes=["background", "liver", "tumor", "vessel"],
    class_map={"background": 0, "liver": 1, "tumor": 2, "vessel": 3},
    
    # Tissue-specific windowing parameters
    window_params={
        "liver": {"window_width": 150, "window_level": 30},
        "tumor": {"window_width": 200, "window_level": 70},
        "vessel": {"window_width": 400, "window_level": 100}
    },
    
    # Advanced data augmentation
    augmentation_params={
        "rotation_range": (-30, 30),
        "zoom_range": (0.9, 1.1),
        "brightness_range": (0.8, 1.2),
        "contrast_range": (0.8, 1.2),
        "elastic_deformation": True,
        "random_flip": True
    }
)

preprocessor = NiiPreprocessor(data_config)
```

### 2. Dataset and DataLoader Module

NII Trainer provides flexible dataset classes that support class balancing and custom transformations.

#### Basic Dataset Creation

```python
from nii_trainer.data import MultiClassSegDataset, PairedTransform
from torch.utils.data import DataLoader

# Create dataset transform
transform = PairedTransform(
    img_size=(512, 512),
    augment=True  # Enable data augmentation
)

# Create training dataset
train_dataset = MultiClassSegDataset(
    data_dir="processed_data/train",
    class_map={"background": 0, "liver": 1, "tumor": 2},
    transform=transform,
    balance=True,  # Balance dataset across classes
    required_classes=[1, 2]  # Focus on liver and tumor classes
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

#### Advanced Data Pipeline

For a complete data pipeline setup, use the utility functions:

```python
from nii_trainer.data import setup_data_pipeline
from nii_trainer.configs.config import DataConfig
from nii_trainer.utils import setup_logger

# Setup logger
logger = setup_logger("dataset_setup", "logs")

# Create data configuration
data_config = DataConfig(
    base_dir="raw_data",
    output_dir="processed_data",
    classes=["background", "liver", "tumor"],
    class_map={"background": 0, "liver": 1, "tumor": 2},
    img_size=(512, 512),
    batch_size=16,
    num_workers=4,
    train_val_test_split=(0.7, 0.15, 0.15),
    balance_dataset=True
)

# Setup complete data pipeline
datasets, dataloaders = setup_data_pipeline(
    config=data_config,
    logger=logger
)

# Use the dataloaders
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]
```

### 3. Model Architecture Module

Configure your model architecture with multiple options for encoders, decoders, and cascade stages.

#### Basic Model Configuration

```python
from nii_trainer.configs.config import TrainerConfig, CascadeConfig, StageConfig

# Create cascade configuration with one stage for liver segmentation
cascade_config = CascadeConfig(
    stages=[
        StageConfig(
            input_classes=["background", "liver"],
            target_class="liver",
            encoder_type="mobilenet_v2",  # Options: mobilenet_v2, resnet18, resnet50, efficientnet
            num_layers=5,
            skip_connections=True,
            dropout_rate=0.3,
            is_binary=True  # Binary segmentation (foreground vs background)
        )
    ],
    in_channels=1,  # Single channel input (grayscale)
    initial_features=32,  # Starting feature count
    pretrained=True  # Use pretrained encoder weights
)

# Create the model
from nii_trainer.models import FlexibleCascadedUNet
model = FlexibleCascadedUNet(cascade_config)
```

#### Multi-Stage Cascade Architecture

Configure a complex multi-stage segmentation cascade:

```python
from nii_trainer.configs.config import CascadeConfig, StageConfig

# Create a 3-stage cascade configuration
cascade_config = CascadeConfig(
    stages=[
        # Stage 1: Binary liver segmentation with EfficientNet
        StageConfig(
            input_classes=["background", "liver"],
            target_class="liver",
            encoder_type="efficientnet",
            num_layers=5,
            skip_connections=True,
            dropout_rate=0.3,
            is_binary=True,
            threshold=0.5  # Prediction threshold
        ),
        # Stage 2: Tumor segmentation within liver using ResNet50
        StageConfig(
            input_classes=["liver"],  # Uses liver mask from previous stage
            target_class="tumor",
            encoder_type="resnet50",
            num_layers=4,
            skip_connections=True,
            dropout_rate=0.4,
            threshold=0.4
        ),
        # Stage 3: Vessel segmentation within liver using MobileNetV2
        StageConfig(
            input_classes=["liver"],  # Uses liver mask from stage 1
            target_class="vessel",
            encoder_type="mobilenet_v2",
            num_layers=3,
            skip_connections=True,
            dropout_rate=0.3,
            threshold=0.5
        )
    ],
    in_channels=1,
    initial_features=64,
    feature_growth=2.0,  # Feature multiplication between stages
    pretrained=True
)
```

### 4. Training Configuration Module

Configure all aspects of model training, from optimizer settings to loss functions.

#### Basic Training Configuration

```python
from nii_trainer.configs.config import TrainingConfig, LossConfig

# Configure training parameters
training_config = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=1e-5,
    epochs=100,
    patience=10,  # Early stopping patience
    mixed_precision=True,  # Enable mixed precision for faster training
    batch_accumulation=2,  # Gradient accumulation steps
    optimizer_type="adam",  # Options: adam, sgd, adamw
    scheduler_type="plateau"  # Options: plateau, step, cosine
)

# Configure loss function
loss_config = LossConfig(
    weight_bce=0.5,  # Binary cross-entropy weight
    weight_dice=0.5,  # Dice loss weight
    focal_gamma=2.0,  # Focal loss gamma parameter
    reward_coef=0.1,  # Reward coefficient for adaptive weighting
    
    # Per-stage loss weighting
    stage_weights=[1.0, 1.2, 1.5],  # Higher weights for later stages
    
    # Per-class weighting to address class imbalance
    class_weights={
        "background": 1.0,
        "liver": 2.0,
        "tumor": 4.0,  # Higher weight for tumor class (typically rarer)
        "vessel": 3.0
    },
    
    # Per-class prediction thresholds
    threshold_per_class=[0.5, 0.4, 0.5]
)
```

#### Complete Trainer Configuration

Combine all configurations into a unified trainer configuration:

```python
from nii_trainer.configs.config import TrainerConfig, DataConfig

# Create complete trainer configuration
trainer_config = TrainerConfig(
    data=data_config,  # Data configuration from previous examples
    cascade=cascade_config,  # Cascade configuration from previous examples
    training=training_config,  # Training configuration from above
    loss=loss_config,  # Loss configuration from above
    experiment_name="liver_multiorgan_advanced",
    save_dir="./experiments"
)
```

### 5. Model Training Module

Configure and run model training with different approaches, including curriculum learning.

#### Standard Training

```python
from nii_trainer.models import ModelTrainer
import torch.optim as optim

# Create trainer
trainer = ModelTrainer(
    config=trainer_config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optim.Adam(model.parameters(), lr=1e-4),
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=5, 
        factor=0.2
    ),
    logger=logger
)

# Train the model
results = trainer.train()
```

#### Curriculum Learning

Train your model using curriculum learning, focusing on one stage at a time:

```python
# Train with curriculum learning
curriculum_params = {
    'stage_schedule': [
        (0, 40),  # Train liver stage for 40 epochs
        (1, 60)   # Add tumor stage for 60 more epochs
    ],
    'learning_rates': [1e-3, 5e-4],  # Different learning rates per stage
    'stage_freezing': [False, True]   # Freeze previous stages when training new ones
}

results = trainer.train_with_curriculum(
    stage_schedule=curriculum_params['stage_schedule'],
    learning_rates=curriculum_params['learning_rates'],
    stage_freezing=curriculum_params['stage_freezing']
)
```

#### Transfer Learning and Fine-Tuning

Load a pre-trained model and fine-tune it for a new task:

```python
# Create a new model with the same architecture
model = FlexibleCascadedUNet(cascade_config)
trainer = ModelTrainer(config, model, train_loader, val_loader)

# Load pre-trained weights
trainer.load_checkpoint("experiments/pretrained_model/best_model.pth")

# Choose a fine-tuning strategy
# Strategy 1: Fine-tune all layers with a smaller learning rate
trainer.optimizer = optim.Adam(model.parameters(), lr=1e-5)
trainer.train(epochs=20)

# Strategy 2: Only fine-tune specific stages
# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False
    
# Unfreeze only the tumor stage parameters
for param in model.stages[1].parameters():
    param.requires_grad = True
    
trainer.optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-5
)
trainer.train(epochs=15)
```

### 6. Visualization and Metrics Module

NII Trainer provides comprehensive tools for visualizing and monitoring training progress and results.

#### Basic Metrics Visualization

```python
from nii_trainer.visualization import SegmentationVisualizer

# Initialize visualizer
visualizer = SegmentationVisualizer(
    class_names=["background", "liver", "tumor"],
    save_dir="visualizations"
)

# Plot training metrics history
visualizer.plot_metrics(
    metrics=trainer.metrics_history,
    save_path="results/training_metrics.png"
)

# Generate confusion matrix
visualizer.plot_confusion_matrix(
    predictions=model_predictions,
    targets=ground_truth,
    save_path="results/confusion_matrix.png"
)
```

#### Advanced Visualization

Visualize model predictions and errors:

```python
# Visualize a batch of predictions
for images, targets in test_loader:
    # Get model predictions
    predictions = model(images.to(device))
    
    # Visualize each sample
    for i in range(min(8, len(images))):
        visualizer.visualize_prediction(
            image=images[i],
            prediction=predictions[i],
            target=targets[i],
            overlay_alpha=0.6,
            show_class_colors=True,
            highlight_errors=True,
            save_path=f"results/sample_{i}.png"
        )
    break

# Find and visualize worst cases (samples with lowest Dice scores)
worst_cases = visualizer.find_worst_cases(
    dataloader=test_loader,
    model=model,
    metric="dice",
    n_cases=5,
    class_name="tumor"
)

for idx, sample in enumerate(worst_cases):
    visualizer.visualize_prediction(
        image=sample["image"],
        prediction=sample["prediction"],
        target=sample["target"],
        metrics=sample["metrics"],
        save_path=f"results/error_analysis/worst_case_{idx}.png"
    )
```

#### Comprehensive Metric Tracking

Monitor and log a wide range of metrics:

```python
# Evaluate model on test set
metrics = trainer.evaluate(test_loader)

# Print results with formatting
print("\nTest Results:")
print("=" * 50)
for key in ["loss", "dice", "precision", "recall", "iou"]:
    if key in metrics:
        print(f"{key.capitalize():15s}: {metrics[key]:.4f}")

# Per-class metrics
for class_name in ["liver", "tumor"]:
    class_key = f"{class_name}_dice"
    if class_key in metrics:
        print(f"{class_name.capitalize():10s} Dice: {metrics[class_key]:.4f}")

# Save metrics to file
import json
with open("results/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
```

### 7. Experiment Workflow Module

NII Trainer's Experiment class provides a unified interface for running complete workflows.

#### Complete Experiment Workflow

```python
from nii_trainer.utils import Experiment, setup_logger
from nii_trainer.configs import create_liver_tumor_config

# Setup logging
logger = setup_logger(
    experiment_name="liver_tumor_workflow",
    save_dir="logs",
    level="INFO"
)

# Create configuration
config = create_liver_tumor_config(
    volume_dir="data/volumes",
    output_dir="data/processed",
    img_size=(512, 512),
    batch_size=16,
    window_width=180,
    window_level=50,
    skip_empty=True,
    slice_step=1,
    train_val_test_split=(0.7, 0.15, 0.15),
    learning_rate=1e-4,
    epochs=100,
    experiment_name="liver_tumor_complete"
)

# Create and initialize experiment
experiment = Experiment(
    config=config,
    experiment_name="liver_tumor_complete",
    base_dir="experiments",
    logger=logger
)

# Run the complete pipeline
results = experiment.run(
    volume_dir="data/volumes",
    segmentation_dir="data/segmentations",
    process_data=True,
    curriculum=True,
    curriculum_params={
        'stage_schedule': [
            (0, 40),  # Liver stage for 40 epochs
            (1, 60)   # Tumor stage for 60 epochs
        ],
        'learning_rates': [1e-3, 5e-4],
        'stage_freezing': [False, True]
    },
    force_overwrite=False
)

# Generate final visualizations
experiment.visualize()

# Print performance summary
experiment.print_metrics(format="table")
```

#### Custom Workflow Components

You can also use the Experiment class for more granular control:

```python
# Create experiment
experiment = Experiment(config=config)

# Step 1: Process data only if needed
if not Path("data/processed").exists():
    experiment.process_data(
        volume_dir="data/volumes",
        segmentation_dir="data/segmentations",
        force_overwrite=False
    )

# Step 2: Setup data pipeline manually
datasets, dataloaders = experiment.setup_data_pipeline()

# Step 3: Setup model and trainer
model, trainer = experiment.setup_model()

# Step 4: Train with custom parameters
experiment.train(
    curriculum=True,
    curriculum_params=custom_curriculum_params
)

# Step 5: Evaluate on specific data
test_metrics = experiment.evaluate(
    dataloader=dataloaders["test"]
)

# Step 6: Generate specific visualizations
experiment.visualize_predictions(
    dataloader=dataloaders["test"],
    num_samples=10,
    save_dir="results/visualizations"
)

# Step 7: Save experiment summary
experiment.save_summary()
```

### 8. Real-world Example: Liver and Tumor Segmentation

Here's a comprehensive example for liver and tumor segmentation:

```python
from nii_trainer.utils import Experiment, setup_logger
from nii_trainer.configs.config import (
    TrainerConfig, DataConfig, CascadeConfig, 
    StageConfig, TrainingConfig, LossConfig
)
from pathlib import Path

# Setup logging
logger = setup_logger("liver_tumor_project", "logs")

# 1. Create detailed configuration
config = TrainerConfig(
    data=DataConfig(
        base_dir="dataset/LiTS",
        output_dir="processed/LiTS",
        classes=["background", "liver", "tumor"],
        class_map={"background": 0, "liver": 1, "tumor": 2},
        img_size=(512, 512),
        batch_size=8,
        num_workers=4,
        window_params={"window_width": 200, "window_level": 40},
        skip_empty=True,
        slice_step=2,
        train_val_test_split=(0.7, 0.15, 0.15),
        balance_dataset=True
    ),
    cascade=CascadeConfig(
        stages=[
            # Stage 1: Binary liver segmentation
            StageConfig(
                input_classes=["background", "liver"],
                target_class="liver",
                encoder_type="efficientnet",
                num_layers=4,
                skip_connections=True,
                dropout_rate=0.3,
                is_binary=True,
                threshold=0.5
            ),
            # Stage 2: Tumor segmentation within liver
            StageConfig(
                input_classes=["liver"],
                target_class="tumor",
                encoder_type="resnet50",
                num_layers=5,
                skip_connections=True,
                dropout_rate=0.4,
                is_binary=True,
                threshold=0.4
            )
        ],
        in_channels=1,
        initial_features=32,
        feature_growth=2.0,
        pretrained=True
    ),
    training=TrainingConfig(
        learning_rate=2e-4,
        weight_decay=1e-5,
        epochs=120,
        patience=15,
        mixed_precision=True,
        batch_accumulation=2,
        optimizer_type="adam",
        scheduler_type="plateau"
    ),
    loss=LossConfig(
        weight_bce=0.4,
        weight_dice=0.6,
        focal_gamma=2.0,
        reward_coef=0.2,
        stage_weights=[1.0, 1.5],
        class_weights={
            "background": 1.0,
            "liver": 2.0,
            "tumor": 5.0
        },
        threshold_per_class=[0.5, 0.4]
    ),
    experiment_name="liver_tumor_segmentation",
    save_dir="experiments"
)

# 2. Create experiment
experiment = Experiment(
    config=config,
    experiment_name="liver_tumor_segmentation",
    base_dir="experiments",
    logger=logger
)

# 3. Run the complete pipeline
results = experiment.run(
    volume_dir="dataset/LiTS/volumes",
    segmentation_dir="dataset/LiTS/segmentations",
    process_data=True,
    curriculum=True,
    curriculum_params={
        'stage_schedule': [
            (0, 60),  # Train liver stage for 60 epochs
            (1, 60)   # Add tumor stage for 60 more epochs
        ],
        'learning_rates': [5e-4, 2e-4],
        'stage_freezing': [False, True]
    },
    force_overwrite=False
)

# 4. Print and save results
experiment.print_metrics()
experiment.save_summary(include_config=True)

# 5. Generate comprehensive visualizations
experiment.visualize(
    save_dir="results/visualizations",
    include_worst_cases=True,
    include_confusion_matrix=True,
    include_metrics_plots=True
)

# 6. Export model for inference
experiment.export_model(
    export_format="onnx",
    export_path="models/liver_tumor_model.onnx"
)
```

## 🧪 Experimental Features

NII Trainer also includes experimental features that are actively being developed:

- **Domain Adaptation**: Adapt models trained on one dataset to perform well on another
- **Uncertainty Estimation**: Quantify model uncertainty using Monte Carlo Dropout
- **Active Learning**: Prioritize annotation of most informative samples
- **Federated Learning**: Train models across multiple sites without sharing data
- **Semi-Supervised Learning**: Leverage unlabeled data to improve segmentation

## 🤝 Contributing

Contributions to NII Trainer are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to development.

### Development Roadmap

- **Version 0.2.0** (Upcoming):
  - Support for 3D models
  - Integration with MONAI framework
  - Additional encoder architectures
  - Performance optimizations

- **Version 0.3.0**:
  - Cloud integration
  - Distributed training support
  - Web-based visualization dashboard
  - Pre-trained model zoo

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