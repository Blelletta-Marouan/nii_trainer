from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
from ..utils.logging_utils import setup_logger, log_config

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
    config = create_liver_config()  # or custom_config
    
    # Setup logging
    logger = setup_logger(
        experiment_name=config.experiment_name,
        save_dir=config.save_dir
    )
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Log configuration
    log_config(logger, {
        'data': vars(config.data),
        'training': vars(config.training),
        'loss': vars(config.loss)
    })
    
    # 2. Create preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    preprocessor = NiiPreprocessor(config.data)
    
    # 3. Create datasets
    logger.info("Creating datasets and dataloaders...")
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
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Create validation dataset
    val_dataset = MultiClassSegDataset(
        data_dir=str(Path(config.data.output_dir) / "val"),
        class_map=config.data.class_map,
        transform=transform_val,
        balance=False
    )
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
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
    logger.info("Creating model...")
    model = FlexibleCascadedUNet(config.cascade)
    model = model.to(config.training.device)
    logger.info(f"Model created and moved to {config.training.device}")
    
    # 5. Create optimizer
    logger.info("Setting up optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 6. Create learning rate scheduler
    logger.info("Setting up learning rate scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.training.reduce_lr_patience,
        factor=0.1,
        verbose=True
    )
    
    # 7. Create trainer
    logger.info("Initializing trainer...")
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger  # Pass logger to trainer
    )
    
    # 8. Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")
    
    # 9. Create visualizer for results
    logger.info("Creating visualizations...")
    visualizer = SegmentationVisualizer(
        class_names=config.data.classes,
        save_dir=str(Path(config.save_dir) / config.experiment_name / "visualizations")
    )
    
    # 10. Visualize some predictions
    logger.info("Generating final predictions visualization...")
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
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()