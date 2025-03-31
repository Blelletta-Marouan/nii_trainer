from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from nii_trainer.configs.config import (
    TrainerConfig,
    StageConfig,
    CascadeConfig,
    DataConfig,
    TrainingConfig,
    LossConfig,
    create_liver_config
)
from nii_trainer.data.preprocessing import NiiPreprocessor
from nii_trainer.data.dataset import MultiClassSegDataset, PairedTransform
from nii_trainer.models.cascaded_unet import FlexibleCascadedUNet
from nii_trainer.models.trainer import ModelTrainer
from nii_trainer.visualization.visualizer import SegmentationVisualizer

def main():
    # 1. Create or load configuration
    # Option 1: Use the predefined liver configuration
    config = create_liver_config()
    
    # Option 2: Create a custom configuration
    # Example: 4-stage cascade for brain tumor segmentation
    custom_config = TrainerConfig(
        data=DataConfig(
            base_dir="/path/to/data",
            output_dir="/path/to/output",
            classes=[
                "background",
                "white_matter",
                "gray_matter",
                "tumor_core",
                "edema"
            ],
            class_map=None,  # Will be auto-generated
            img_size=(256, 256),
            batch_size=8,
            window_params={
                "width": 80,
                "level": 40
            }
        ),
        cascade=CascadeConfig(
            stages=[
                # Stage 1: Segment brain tissue from background
                StageConfig(
                    input_classes=["background", "white_matter", "gray_matter"],
                    target_class="white_matter",
                    encoder_type="resnet18",
                    encoder_layers=5,
                    decoder_layers=5
                ),
                # Stage 2: Segment gray matter
                StageConfig(
                    input_classes=["white_matter", "gray_matter"],
                    target_class="gray_matter",
                    encoder_type="mobilenet_v2",
                    encoder_layers=4,
                    decoder_layers=4
                ),
                # Stage 3: Segment tumor core
                StageConfig(
                    input_classes=["white_matter", "gray_matter", "tumor_core"],
                    target_class="tumor_core",
                    encoder_type="efficientnet",
                    encoder_layers=4,
                    decoder_layers=4
                ),
                # Stage 4: Segment edema
                StageConfig(
                    input_classes=["tumor_core", "edema"],
                    target_class="edema",
                    encoder_type="resnet18",
                    encoder_layers=3,
                    decoder_layers=3
                )
            ],
            in_channels=1,
            initial_features=32,
            feature_growth=2.0
        ),
        training=TrainingConfig(
            learning_rate=1e-4,
            epochs=100,
            batch_accumulation=2  # Effective batch size = batch_size * batch_accumulation
        ),
        loss=LossConfig(
            stage_weights=[1.0, 1.2, 1.5, 1.5],
            class_weights={
                "background": 1.0,
                "white_matter": 2.0,
                "gray_matter": 2.0,
                "tumor_core": 4.0,
                "edema": 3.0
            }
        ),
        experiment_name="brain_tumor_cascade"
    )
    
    # Choose which configuration to use
    config = custom_config  # or config for liver segmentation
    
    # 2. Create preprocessing pipeline
    preprocessor = NiiPreprocessor(config.data)
    
    # 3. Create datasets
    transform_train = PairedTransform(
        image_size=config.data.img_size,
        augment=True
    )
    transform_val = PairedTransform(
        image_size=config.data.img_size,
        augment=False
    )
    
    # Create training dataset
    train_dataset = MultiClassSegDataset(
        data_dir=str(Path(config.data.output_dir) / "train"),
        class_map=config.data.class_map,
        transform=transform_train,
        balance=config.data.balance_dataset
    )
    
    # Create validation dataset
    val_dataset = MultiClassSegDataset(
        data_dir=str(Path(config.data.output_dir) / "val"),
        class_map=config.data.class_map,
        transform=transform_val,
        balance=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # 4. Create model
    model = FlexibleCascadedUNet(config.cascade)
    model = model.to(config.training.device)
    
    # 5. Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 6. Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.training.reduce_lr_patience,
        factor=0.1,
        verbose=True
    )
    
    # 7. Create trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # 8. Train model
    trainer.train()
    
    # 9. Create visualizer for results
    visualizer = SegmentationVisualizer(
        class_names=config.data.classes,
        save_dir=str(Path(config.save_dir) / config.experiment_name / "visualizations")
    )
    
    # 10. Visualize some predictions
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(config.training.device)
            outputs = model(images)
            predictions = model.get_predictions(
                outputs,
                [stage.threshold for stage in config.cascade.stages]
            )
            
            # Visualize batch
            visualizer.visualize_batch(
                images.cpu(),
                predictions.cpu(),
                targets,
                max_samples=4,
                save_dir=str(Path(config.save_dir) / config.experiment_name / "final_predictions")
            )
            break  # Just visualize one batch

if __name__ == "__main__":
    main()