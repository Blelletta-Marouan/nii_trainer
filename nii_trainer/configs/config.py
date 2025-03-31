from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import torch

@dataclass
class StageConfig:
    """Configuration for a single stage in the cascaded model."""
    input_classes: List[str]  # Classes to consider as input
    target_class: str        # Class to segment in this stage
    encoder_type: str = "mobilenet_v2"  # Encoder backbone for this stage
    encoder_layers: int = 5  # Number of encoder layers
    decoder_layers: int = 5  # Number of decoder layers
    skip_connections: bool = True
    dropout_rate: float = 0.3
    threshold: float = 0.5   # Prediction threshold for this stage
    is_binary: bool = False  # Whether this stage performs binary segmentation

@dataclass
class CascadeConfig:
    """Configuration for the entire cascade of models."""
    stages: List[StageConfig]
    in_channels: int = 1
    initial_features: int = 32  # Initial number of features
    feature_growth: float = 2.0  # Feature multiplication factor between stages
    pretrained: bool = True

@dataclass
class DataConfig:
    base_dir: str
    output_dir: str
    classes: List[str]  # All possible classes in order of segmentation
    class_map: Dict[str, int]  # Mapping of class names to indices
    img_size: Tuple[int, int] = (512, 512)
    batch_size: int = 16
    num_workers: int = 4
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    slice_step: int = 1
    skip_empty: bool = True
    balance_dataset: bool = True
    window_params: Dict[str, Any] = None  # CT window parameters per class if needed
    augmentation_params: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-8
    epochs: int = 60
    patience: int = 11
    reduce_lr_patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    optimizer_type: str = "adam"
    scheduler_type: str = "plateau"
    batch_accumulation: int = 1

@dataclass
class LossConfig:
    """Loss configuration with per-stage settings."""
    weight_bce: float = 0.5
    weight_dice: float = 0.5
    focal_gamma: float = 2.0
    reward_coef: float = 0.1
    stage_weights: List[float] = None  # Weight for each stage's loss
    class_weights: Dict[str, float] = None  # Weight for each class

@dataclass
class TrainerConfig:
    """Main configuration class combining all configs."""
    data: DataConfig
    cascade: CascadeConfig
    training: TrainingConfig
    loss: LossConfig
    experiment_name: str
    save_dir: str = "./experiments"

    def __post_init__(self):
        # Validate class configuration
        if not self.data.classes:
            raise ValueError("Must specify at least one class")
            
        # Create class mapping if not provided
        if not self.data.class_map:
            self.data.class_map = {cls: idx for idx, cls in enumerate(self.data.classes)}
            
        # Set default stage weights if not provided
        if not self.loss.stage_weights:
            self.loss.stage_weights = [1.0] * len(self.cascade.stages)
            
        # Set default class weights if not provided
        if not self.loss.class_weights:
            self.loss.class_weights = {cls: 1.0 for cls in self.data.classes}
            
        # Validate stage configurations
        self._validate_cascade_config()
    
    def _validate_cascade_config(self):
        """Validate cascade configuration for consistency."""
        # Check that all classes mentioned in stages exist in data.classes
        all_stage_classes = set()
        for stage in self.cascade.stages:
            all_stage_classes.update(stage.input_classes)
            all_stage_classes.add(stage.target_class)
            
        unknown_classes = all_stage_classes - set(self.data.classes)
        if unknown_classes:
            raise ValueError(f"Unknown classes in stage configs: {unknown_classes}")
            
        # Validate stage progression
        processed_classes = set()
        for i, stage in enumerate(self.cascade.stages):
            # Check that input classes for this stage have been processed
            unprocessed = set(stage.input_classes) - processed_classes - {stage.target_class}
            if unprocessed and i > 0:  # First stage can take any classes
                raise ValueError(f"Stage {i} requires unprocessed classes: {unprocessed}")
            processed_classes.add(stage.target_class)

def create_liver_config() -> TrainerConfig:
    """Create a sample configuration for liver segmentation."""
    # Define all possible classes
    classes = ["background", "liver", "tumor", "vessel", "nerve"]
    
    # Define cascade stages
    stages = [
        # First stage: Binary segmentation (foreground vs background)
        StageConfig(
            input_classes=["background", "foreground"],
            target_class="foreground",
            encoder_type="mobilenet_v2",
            encoder_layers=5,
            decoder_layers=5,
            is_binary=True
        ),
        # Subsequent stages: Fine-grained segmentation
        StageConfig(
            input_classes=["foreground"],
            target_class="liver",
            encoder_type="resnet18",
            encoder_layers=4,
            decoder_layers=4
        ),
        StageConfig(
            input_classes=["liver"],
            target_class="tumor",
            encoder_type="efficientnet",
            encoder_layers=4,
            decoder_layers=4
        ),
        StageConfig(
            input_classes=["liver"],
            target_class="vessel",
            encoder_type="mobilenet_v2",
            encoder_layers=3,
            decoder_layers=3
        )
    ]
    
    return TrainerConfig(
        data=DataConfig(
            base_dir="path/to/data",
            output_dir="path/to/output",
            classes=classes,
            class_map={cls: idx for idx, cls in enumerate(classes)},
            window_params={"width": 180, "level": 50}
        ),
        cascade=CascadeConfig(
            stages=stages,
            in_channels=1,
            initial_features=32,
            feature_growth=2.0
        ),
        training=TrainingConfig(),
        loss=LossConfig(
            stage_weights=[1.0, 1.2, 1.2, 1.2],
            class_weights={
                "background": 1.0,
                "liver": 2.0,
                "tumor": 4.0,
                "vessel": 3.0,
                "nerve": 3.0
            }
        ),
        experiment_name="liver_multiorgan_seg"
    )