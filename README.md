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

For a more detailed approach:

```python
from nii_trainer.configs import create_liver_config
from nii_trainer.models import FlexibleCascadedUNet, ModelTrainer
from nii_trainer.data import NiiPreprocessor, MultiClassSegDataset
from torch.utils.data import DataLoader
import torch.optim as optim

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
val_dataset = MultiClassSegDataset(
    data_dir="processed_data/val",
    class_map=config.data.class_map
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 4. Initialize model and optimizer
model = FlexibleCascadedUNet(config.cascade)
optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# 5. Initialize trainer and train
trainer = ModelTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler
)

# 6. Train with standard approach
trainer.train()

# OR train with curriculum learning
trainer.train_with_curriculum()
```

## 📋 Comprehensive Guide

### 1. Data Preparation and Preprocessing

NII Trainer provides comprehensive tools for preparing medical imaging data:

#### Standard Preprocessing Workflow

```python
from nii_trainer.data import NiiPreprocessor
from nii_trainer.configs.config import DataConfig
from pathlib import Path

# Basic configuration for preprocessing
data_config = DataConfig(
    base_dir="raw_data",
    output_dir="processed_data",
    classes=["background", "liver", "tumor"],
    class_map={"background": 0, "liver": 1, "tumor": 2},
    img_size=(512, 512),
    batch_size=16,
    window_params={"window_width": 180, "window_level": 50},
    skip_empty=True,
    slice_step=1
)

# Initialize preprocessor
preprocessor = NiiPreprocessor(data_config)

# Process a single volume-segmentation pair
preprocessor.extract_slices(
    volume_path="raw_data/volume-0.nii",
    segmentation_path="segmentations/segmentation-0.nii",
    output_dir=Path("processed_data")
)

# Process multiple volumes
volume_paths = sorted(Path("volume_pt1").glob("*.nii"))
segmentation_paths = sorted(Path("segmentations").glob("*.nii"))

for vol_path, seg_path in zip(volume_paths[:10], segmentation_paths[:10]):
    preprocessor.extract_slices(
        volume_path=str(vol_path),
        segmentation_path=str(seg_path),
        output_dir=Path("processed_data")
    )
```

#### Advanced Preprocessing Options

```python
# Advanced preprocessing with custom window parameters for different tissues
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
        # Specialized windowing for different tissue types
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
)

# Processing with slice selection based on content
for volume, segmentation in data_pairs:
    # Extract metadata from NIfTI files to determine appropriate slices
    metadata = read_nii_metadata(volume)
    
    # Select slices based on custom criteria
    # For example, focus on slices with high tumor content
    selected_slices = select_slices_with_criteria(
        segmentation, 
        target_class=2,  # tumor class
        min_area_percent=5.0,  # at least 5% tumor area
        max_slices=50  # limit to 50 slices
    )
    
    # Process selected slices
    preprocessor.extract_slices(
        volume_path=volume,
        segmentation_path=segmentation,
        output_dir="processed_data",
        slice_indices=selected_slices
    )
```

### 2. Model Configuration and Architecture Customization

NII Trainer provides extensive configuration options for model architecture:

#### Basic Configuration

```python
from nii_trainer.configs.config import TrainerConfig, DataConfig, CascadeConfig, StageConfig

# Basic configuration for liver segmentation
basic_config = TrainerConfig(
    data=DataConfig(
        base_dir="data",
        output_dir="output",
        classes=["background", "liver"],
        class_map={"background": 0, "liver": 1},
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
    ),
    training=TrainingConfig(
        learning_rate=1e-3,
        epochs=50
    ),
    loss=LossConfig(
        weight_bce=0.5,
        weight_dice=0.5
    ),
    experiment_name="liver_segmentation"
)
```

#### Advanced Multi-Stage Configuration

```python
# Advanced multi-stage configuration for liver-tumor-vessel segmentation
advanced_config = TrainerConfig(
    data=DataConfig(
        base_dir="data",
        output_dir="output",
        classes=["background", "liver", "tumor", "vessel"],
        class_map={"background": 0, "liver": 1, "tumor": 2, "vessel": 3},
        img_size=(512, 512),
        window_params={"window_width": 180, "window_level": 50},
        batch_size=16,
        train_val_test_split=(0.7, 0.15, 0.15)
    ),
    cascade=CascadeConfig(
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
                threshold=0.5
            ),
            # Stage 2: Tumor segmentation within liver using ResNet50
            StageConfig(
                input_classes=["liver"],
                target_class="tumor",
                encoder_type="resnet50",
                num_layers=4,
                skip_connections=True,
                dropout_rate=0.4,
                threshold=0.4
            ),
            # Stage 3: Vessel segmentation within liver using MobileNetV2
            StageConfig(
                input_classes=["liver"],
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
        feature_growth=2.0,
        pretrained=True
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        epochs=100,
        patience=10,
        reduce_lr_patience=3,
        mixed_precision=True,
        batch_accumulation=2,
        optimizer_type="adam",
        scheduler_type="plateau"
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
        },
        threshold_per_class=[0.5, 0.4, 0.5]
    ),
    experiment_name="liver_multiorgan_advanced"
)
```

#### Creating Custom Model Configurations

```python
# Creating a configuration with custom model architecture
from nii_trainer.configs import create_custom_config

custom_config = create_custom_config(
    # Data configuration
    classes=["background", "organ1", "organ2", "lesion1", "lesion2"],
    img_size=(384, 384),
    batch_size=8,
    window_params={"window_width": 200, "window_level": 60},
    
    # Cascade configuration
    stages=[
        {
            "input": ["background", "organ1", "organ2"],
            "target": "organ1",
            "encoder": "resnet18",
            "layers": 4,
            "is_binary": True
        },
        {
            "input": ["background", "organ1", "organ2"],
            "target": "organ2",
            "encoder": "resnet18",
            "layers": 4,
            "is_binary": True
        },
        {
            "input": ["organ1"],
            "target": "lesion1",
            "encoder": "efficientnet",
            "layers": 3,
            "is_binary": True
        },
        {
            "input": ["organ2"],
            "target": "lesion2",
            "encoder": "mobilenet_v2",
            "layers": 3,
            "is_binary": True
        }
    ],
    
    # Training configuration
    learning_rate=5e-4,
    epochs=120,
    mixed_precision=True,
    
    # Loss configuration
    loss_weights={
        "bce": 0.4,
        "dice": 0.6,
        "focal_gamma": 2.0,
        "reward": 0.2
    },
    class_weights={
        "organ1": 2.0,
        "organ2": 2.0,
        "lesion1": 5.0,
        "lesion2": 4.0
    }
)
```

### 3. Training Strategies and Advanced Features

NII Trainer offers various training strategies to optimize model performance:

#### Standard Training

```python
# Standard training approach
trainer.train(
    epochs=100,
    early_stopping=True,
    patience=10,
    checkpoint_interval=5,
    validate_interval=1,
    visualize_interval=10,
    save_best_only=True,
    verbose=1
)
```

#### Curriculum Learning

```python
# Curriculum learning with stage-wise training
trainer.train_with_curriculum(
    # Stage schedule defines when to add each stage and for how many epochs
    stage_schedule=[
        (0, 30),   # Train liver stage for 30 epochs
        (1, 30),   # Add tumor stage for 30 more epochs
        (2, 40)    # Add vessel stage for 40 more epochs
    ],
    # Different learning rates for each stage
    learning_rates=[1e-3, 5e-4, 1e-4],
    # Control whether to freeze previous stages
    stage_freezing=[False, True, True],
    # Optional: different optimizers for different stages
    stage_optimizers=["adam", "adam", "sgd"],
    # Optional: different learning rate schedules
    lr_schedules=[
        {"type": "step", "step_size": 10, "gamma": 0.5},
        {"type": "plateau", "patience": 5, "factor": 0.2},
        {"type": "cosine", "T_max": 40}
    ],
    # Early stopping parameters
    early_stopping=True,
    patience=15,
    # Checkpoint settings
    save_best_only=True,
    checkpoint_interval=5
)
```

#### Transfer Learning and Fine-Tuning

```python
# Load a pre-trained model and fine-tune on new data
from nii_trainer.models import FlexibleCascadedUNet, ModelTrainer

# Create new model with same architecture as pre-trained model
model = FlexibleCascadedUNet(config.cascade)

# Initialize trainer
trainer = ModelTrainer(config, model, train_loader, val_loader)

# Load pre-trained weights
trainer.load_checkpoint("experiments/pretrained_model/best_model.pth")

# Fine-tune with different strategies
trainer.fine_tune(
    # Strategy 1: Fine-tune all layers with a smaller learning rate
    mode="all",
    learning_rate=1e-5,
    epochs=20,
    
    # OR Strategy 2: Only fine-tune specific stages
    # mode="stages",
    # stages=[1, 2],  # Only fine-tune tumor and vessel stages
    # learning_rate=2e-5,
    # epochs=15,
    
    # OR Strategy 3: Progressive unfreezing
    # mode="progressive",
    # epochs_per_stage=10,
    # learning_rates=[1e-5, 2e-5, 5e-5],
    # stages=[2, 1, 0],  # Unfreeze in reverse order
    
    # Common parameters
    early_stopping=True,
    patience=5,
    save_best_only=True
)
```

#### Advanced Training Features

```python
# Implement mixed precision training and gradient accumulation
trainer = ModelTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    mixed_precision=True,  # Enable mixed precision training
    gradient_accumulation_steps=4  # Accumulate gradients over 4 batches
)

# Monitor and adjust training dynamically
class CustomTrainingCallback:
    def on_epoch_end(self, epoch, metrics):
        # Adjust class weights based on performance
        if metrics["val_dice_tumor"] < 0.6 and epoch > 10:
            trainer.update_class_weights({"tumor": 5.0})
        
        # Add data augmentation after certain epochs
        if epoch == 20:
            trainer.add_augmentation(
                rotation_range=(-45, 45),
                elastic_deformation=True
            )
            
        # Print custom metrics
        print(f"Epoch {epoch}: Tumor/Liver Dice Ratio: "
              f"{metrics['val_dice_tumor'] / metrics['val_dice_liver']:.3f}")

# Register callback
trainer.register_callback(CustomTrainingCallback())

# Train with the callback
trainer.train()
```

### 4. Evaluation, Visualization and Analysis

NII Trainer provides comprehensive tools for evaluating and visualizing segmentation results:

#### Basic Evaluation

```python
# Basic evaluation on test dataset
test_metrics = trainer.evaluate(test_loader)

# Print results
for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")

# Save detailed metrics to CSV and JSON
trainer.save_metrics("results/test_metrics.csv")
trainer.save_metrics("results/test_metrics.json", format="json")
```

#### Advanced Visualization

```python
from nii_trainer.visualization import SegmentationVisualizer
import matplotlib.pyplot as plt

# Initialize visualizer
visualizer = SegmentationVisualizer(
    class_names=config.data.classes,
    save_dir="visualizations"
)

# Visualize a batch of predictions
for images, targets in test_loader:
    # Get model predictions
    predictions = model(images.to(config.training.device))
    
    # Visualize with different options
    visualizer.visualize_batch(
        images=images,
        predictions=predictions,
        targets=targets,
        max_samples=8,
        overlay_alpha=0.6,
        show_uncertainty=True,
        save_path="results/batch_visualization.png"
    )
    break  # Just visualize one batch

# Visualize worst predictions (cases with lowest Dice score)
worst_cases = visualizer.find_worst_cases(
    dataloader=test_loader,
    model=model,
    metric="dice",
    n_cases=5,
    class_name="tumor"
)

# Visualize these cases
visualizer.visualize_cases(
    cases=worst_cases,
    save_dir="results/error_analysis"
)

# Plot metrics history
plt.figure(figsize=(15, 10))

# Create subplots for different metrics
plt.subplot(2, 2, 1)
visualizer.plot_metrics(
    {"Train Loss": trainer.metrics_history["train_loss"],
     "Val Loss": trainer.metrics_history["val_loss"]},
    title="Loss Curves",
    xlabel="Epoch",
    ylabel="Loss"
)

plt.subplot(2, 2, 2)
visualizer.plot_metrics(
    {"Liver Dice": [m.get("val_dice_liver", 0) for m in trainer.metrics_history["val_metrics"]],
     "Tumor Dice": [m.get("val_dice_tumor", 0) for m in trainer.metrics_history["val_metrics"]]},
    title="Dice Scores",
    xlabel="Epoch",
    ylabel="Dice"
)

plt.subplot(2, 2, 3)
visualizer.plot_metrics(
    {"Learning Rate": trainer.metrics_history["learning_rate"]},
    title="Learning Rate",
    xlabel="Epoch",
    ylabel="LR",
    yscale="log"
)

plt.subplot(2, 2, 4)
visualizer.plot_confusion_matrix(
    predictions=model.get_predictions(test_loader),
    targets=get_targets(test_loader),
    class_names=config.data.classes[1:],  # Exclude background
    title="Confusion Matrix"
)

plt.tight_layout()
plt.savefig("results/training_summary.png", dpi=300)
plt.close()
```

#### Volumetric Evaluation

```python
from nii_trainer.evaluation import VolumetricEvaluator

# Initialize volumetric evaluator
volumetric_evaluator = VolumetricEvaluator(
    model=model,
    dataset=test_3d_dataset,
    class_names=config.data.classes,
    device=config.training.device,
    spacing=(1.0, 1.0, 2.5)  # Physical spacing in mm
)

# Evaluate volumetric metrics (Dice, Hausdorff distance, etc.)
volume_metrics = volumetric_evaluator.evaluate(
    metrics=["dice", "hausdorff", "assd", "volume_similarity"],
    per_case=True,
    save_results=True,
    save_path="results/volumetric_metrics.csv"
)

# Print average results
print("Average Volumetric Metrics:")
for class_name in config.data.classes[1:]:  # Skip background
    print(f"{class_name}:")
    for metric, value in volume_metrics["avg"][class_name].items():
        print(f"  {metric}: {value:.4f}")

# Generate 3D visualizations
volumetric_evaluator.generate_3d_visualizations(
    case_indices=[0, 5, 10],  # Visualize specific cases
    save_dir="results/3d_visualizations",
    render_type="surface",
    include_errors=True
)
```

### 5. Complete Workflow Example: Liver and Tumor Segmentation

Here's a complete workflow example from preprocessing to evaluation:

```python
from nii_trainer.utils import setup_logger, Experiment
from nii_trainer.configs import create_liver_tumor_config

# 1. Setup logging
logger = setup_logger(
    experiment_name="liver_tumor_segmentation",
    save_dir="logs",
    level="INFO"
)

# 2. Create configuration
config = create_liver_tumor_config(
    volume_dir="data/volumes",
    output_dir="data/processed",
    img_size=(512, 512),
    batch_size=16,
    window_width=180,
    window_level=50,
    skip_empty=False,
    slice_step=1,
    train_val_test_split=(0.7, 0.15, 0.15),
    learning_rate=1e-4,
    epochs=100,
    experiment_name="liver_tumor_complete"
)

# 3. Create and initialize experiment
experiment = Experiment(
    config=config,
    experiment_name="liver_tumor_complete",
    base_dir="experiments",
    logger=logger
)

# 4. Process NIfTI data
if not Path("data/processed").exists():
    experiment.process_data(
        volume_dir="data/volumes",
        segmentation_dir="data/segmentations",
        force_overwrite=False
    )

# 5. Setup data pipeline
datasets, dataloaders = experiment.setup_data_pipeline()

# 6. Setup model, optimizer and trainer
model, trainer = experiment.setup_model()

# 7. Train with curriculum learning
curriculum_params = {
    'stage_schedule': [
        (0, 40),  # Train liver stage for 40 epochs
        (1, 60)   # Add tumor stage for 60 more epochs
    ],
    'learning_rates': [1e-3, 5e-4],
    'stage_freezing': [False, True]
}

experiment.train(
    curriculum=True,
    curriculum_params=curriculum_params
)

# 8. Evaluate model on test set
metrics = experiment.evaluate()

# 9. Generate visualizations
experiment.visualize()

# 10. Save experiment summary
experiment.save_summary()

# 11. Print final metrics
logger.info("Final Test Metrics:")
for key, value in metrics.items():
    logger.info(f"  {key}: {value:.4f}")
```

## 🧪 Experimental Features

NII Trainer also includes experimental features that are actively being developed:

- **Domain Adaptation**: Adapt models trained on one dataset to perform well on another
- **Uncertainty Estimation**: Quantify model uncertainty using Monte Carlo Dropout
- **Active Learning**: Prioritize annotation of most informative samples
- **Federated Learning**: Train models across multiple sites without sharing data
- **Semi-Supervised Learning**: Leverage unlabeled data to improve segmentation

Example of uncertainty estimation:

```python
from nii_trainer.experimental import UncertaintyEstimator

# Initialize uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(
    model=model,
    dropout_rate=0.5,
    n_samples=10,
    device=config.training.device
)

# Generate uncertainty maps
for images, targets in test_loader:
    # Get predictions with uncertainty
    predictions, uncertainty = uncertainty_estimator.predict_with_uncertainty(images)
    
    # Visualize
    visualizer.visualize_prediction(
        image=images[0],
        prediction=predictions[0],
        target=targets[0],
        uncertainty=uncertainty[0],
        save_path="results/uncertainty_example.png"
    )
    break
```

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