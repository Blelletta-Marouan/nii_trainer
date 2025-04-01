from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
from nii_trainer.utils.logging_utils import setup_logger

from nii_trainer.configs.config import (
    TrainerConfig,
    StageConfig,
    CascadeConfig,
    DataConfig,
    TrainingConfig,
    LossConfig
)
from nii_trainer.data.preprocessing import NiiPreprocessor
from nii_trainer.data.dataset import MultiClassSegDataset, PairedTransform
from nii_trainer.models.cascaded_unet import FlexibleCascadedUNet
from nii_trainer.models.trainer import ModelTrainer
from nii_trainer.visualization.visualizer import SegmentationVisualizer

# Setup main logger with proper configuration
logger = setup_logger(
    experiment_name="liver_tumor_training",
    save_dir=".",
    level=logging.INFO
)

def main():
    logger.info("=== Starting Liver-Tumor Segmentation Pipeline ===")
    
    # 1. Create configuration for liver-tumor cascade
    logger.info("Creating model configuration...")
    config = TrainerConfig(
        data=DataConfig(
            base_dir=str(Path("volume_pt1").absolute()),
            output_dir="processed_data",
            classes=["background", "liver", "tumor"],
            class_map={"background": 0, "liver": 1, "tumor": 2},
            img_size=(512, 512),
            batch_size=8,
            window_params={"window_width": 180, "window_level": 50},
            skip_empty=True,
            slice_step=4
        ),
        cascade=CascadeConfig(
            stages=[
                # Stage 1: Binary liver segmentation
                StageConfig(
                    input_classes=["background", "liver"],
                    target_class="liver",
                    encoder_type="resnet18",
                    encoder_layers=5,
                    decoder_layers=5,
                    is_binary=True,
                    threshold=0.5
                ),
                # Stage 2: Tumor segmentation within liver
                StageConfig(
                    input_classes=["liver"],  # Only one input class
                    target_class="tumor",
                    encoder_type="efficientnet",
                    encoder_layers=4,
                    decoder_layers=4,
                    is_binary=True,  # Tumor is also binary within liver
                    threshold=0.5
                )
            ],
            in_channels=1,
            initial_features=32,
            feature_growth=2.0,
            pretrained=True
        ),
        training=TrainingConfig(
            learning_rate=1e-4,
            epochs=100,
            patience=10,
            mixed_precision=True
        ),
        loss=LossConfig(
            stage_weights=[1.0, 1.5],  # Higher weight for tumor stage
            class_weights={
                "background": 1.0,
                "liver": 2.0,
                "tumor": 4.0  # Higher weight for tumor class due to small size
            }
        ),
        experiment_name="liver_tumor_cascade"
    )
    
    # 2. Create preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    preprocessor = NiiPreprocessor(config.data)
    
    # Process all volume-segmentation pairs
    train_pairs = []
    volume_dir = Path("volume_pt1")
    segmentation_dir = Path("segmentations")
    
    # Process volumes and their corresponding segmentations
    volume_files = sorted(list(volume_dir.glob("*.nii")))
    logger.info(f"Found {len(volume_files)} volumes to process")
    
    for volume_path in volume_files:
        logger.info(f"Processing volume: {volume_path.name}")
        # Find corresponding segmentation file
        seg_name = f"segmentation-{volume_path.stem.split('-')[-1]}.nii"
        seg_path = segmentation_dir / seg_name
        if seg_path.exists():
            logger.info(f"Found matching segmentation: {seg_path.name}")
            pairs = preprocessor.extract_slices(
                volume_path=str(volume_path),
                segmentation_path=str(seg_path),
                output_dir=Path(config.data.output_dir)
            )
            train_pairs.extend(pairs)
            logger.info(f"Extracted {len(pairs)} valid slices from volume")
    
    logger.info(f"Total number of processed image pairs: {len(train_pairs)}")
    
    # 3. Create datasets
    logger.info("Creating training and validation datasets...")
    num_train = int(0.8 * len(train_pairs))
    train_pairs, val_pairs = train_pairs[:num_train], train_pairs[num_train:]
    logger.info(f"Split data into {len(train_pairs)} training and {len(val_pairs)} validation samples")
    
    transform_train = PairedTransform(
        img_size=config.data.img_size,
        augment=True
    )
    transform_val = PairedTransform(
        img_size=config.data.img_size,
        augment=False
    )
    
    logger.info("Initializing datasets...")
    train_dataset = MultiClassSegDataset(
        data_dir=str(Path(config.data.output_dir) / "train"),
        class_map=config.data.class_map,
        transform=transform_train,
        balance=True
    )
    
    val_dataset = MultiClassSegDataset(
        data_dir=str(Path(config.data.output_dir) / "val"),
        class_map=config.data.class_map,
        transform=transform_val,
        balance=False
    )
    
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 4. Create and train model
    logger.info("Initializing cascaded UNet model...")
    model = FlexibleCascadedUNet(config.cascade)
    model = model.to(config.training.device)
    
    # Create optimizer
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer
    )
    
    # 5. Train with curriculum learning
    logger.info("Starting curriculum training...")
    trainer.train(
        stage_schedule=[
            (0, 50),  # Train liver stage for 50 epochs
            (1, 50)   # Add tumor stage for 50 more epochs
        ],
        learning_rates=[1e-3, 5e-4],
        stage_freezing=[False, True]  # Freeze liver stage when training tumor stage
    )
    
    # 6. Create visualizer for results
    logger.info("Setting up visualization...")
    visualizer = SegmentationVisualizer(
        class_names=config.data.classes,
        save_dir="visualizations"
    )
    
    # Generate some validation predictions
    logger.info("Generating validation predictions for visualization...")
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            predictions = model.get_predictions(outputs)
            
            # Visualize batch
            visualizer.visualize_batch(
                images.cpu(),
                predictions.cpu(),
                targets,
                max_samples=4,
                save_dir="final_predictions"
            )
            logger.info("Saved visualization samples in 'final_predictions' directory")
            break
    
    logger.info("=== Training pipeline completed successfully ===")

if __name__ == "__main__":
    main()