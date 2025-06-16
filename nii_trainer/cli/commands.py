"""
CLI command implementations for NII-Trainer.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict
import argparse

from ..api.high_level import NIITrainer, quick_train, quick_evaluate
from ..core.config import GlobalConfig, create_default_config, load_config
from ..utils.device import get_system_info, get_gpu_info
from ..utils.logging import setup_logging
from ..core.exceptions import NIITrainerError


def train_command(args: argparse.Namespace) -> int:
    """Handle training command."""
    print("🚀 Starting NII-Trainer Training")
    print("=" * 50)
    
    try:
        # Load or create configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        # Override config with command line arguments
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        
        if args.epochs:
            config.training.max_epochs = args.epochs
        
        if args.batch_size:
            config.data.batch_size = args.batch_size
        
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        
        if args.device != 'auto':
            config.device = args.device
        
        if args.mixed_precision:
            config.training.mixed_precision = True
        
        # Set data paths
        config.data.train_data_path = args.train_data
        if args.val_data:
            config.data.val_data_path = args.val_data
        
        config.output_dir = args.output_dir
        
        # Create trainer
        trainer = NIITrainer(
            config=config,
            experiment_name=args.experiment_name or "cli_training",
            output_dir=args.output_dir
        )
        
        # Setup training
        trainer.setup(args.train_data, args.val_data)
        
        print(f"📁 Training data: {args.train_data}")
        print(f"📁 Validation data: {args.val_data or 'Not provided'}")
        print(f"💾 Output directory: {args.output_dir}")
        print(f"🎯 Device: {config.device}")
        print(f"⚡ Mixed precision: {getattr(config.training, 'mixed_precision', False)}")
        print("=" * 50)
        
        # Start training
        results = trainer.train(resume_from=args.resume)
        
        print("\n✅ Training completed successfully!")
        print("📊 Final Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        
        # Save final checkpoint
        checkpoint_path = Path(args.output_dir) / "final_model.pth"
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"💾 Model saved to: {checkpoint_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """Handle evaluation command."""
    print("📊 Starting NII-Trainer Evaluation")
    print("=" * 50)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        # Update evaluation config
        config.evaluation.metrics = args.metrics
        config.evaluation.save_predictions = args.save_predictions
        
        if args.device != 'auto':
            config.device = args.device
        
        # Create evaluator
        evaluator = NIITrainer(
            config=config,
            experiment_name="cli_evaluation",
            output_dir=args.output_dir
        )
        
        # Setup and evaluate
        evaluator.setup(args.test_data)
        
        print(f"📁 Test data: {args.test_data}")
        print(f"🏋️ Model checkpoint: {args.checkpoint}")
        print(f"📊 Metrics: {', '.join(args.metrics)}")
        print(f"💾 Output directory: {args.output_dir}")
        print("=" * 50)
        
        results = evaluator.evaluate(
            test_data_path=args.test_data,
            checkpoint_path=args.checkpoint
        )
        
        print("\n✅ Evaluation completed successfully!")
        print("📊 Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Save results
        results_path = Path(args.output_dir) / "evaluation_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


def predict_command(args: argparse.Namespace) -> int:
    """Handle prediction command."""
    print("🔮 Starting NII-Trainer Prediction")
    print("=" * 50)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        if args.device != 'auto':
            config.device = args.device
        
        # Create predictor
        predictor = NIITrainer(
            config=config,
            experiment_name="cli_prediction"
        )
        
        # Setup
        predictor.setup(args.input)  # Dummy setup
        
        print(f"📁 Input: {args.input}")
        print(f"📁 Output: {args.output}")
        print(f"🏋️ Model checkpoint: {args.checkpoint}")
        print(f"🎯 Device: {config.device}")
        print(f"🔄 Test-time augmentation: {args.tta}")
        print("=" * 50)
        
        # Check if input is file or directory
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if input_path.is_file():
            # Single file prediction
            prediction = predictor.predict(
                image_path=str(input_path),
                output_path=str(output_path),
                checkpoint_path=args.checkpoint
            )
            print(f"✅ Prediction saved to: {output_path}")
            
        elif input_path.is_dir():
            # Batch prediction
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find medical image files
            image_extensions = {'.nii', '.nii.gz', '.dcm', '.nrrd'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.rglob(f'*{ext}'))
            
            print(f"🔍 Found {len(image_files)} medical images")
            
            for i, image_file in enumerate(image_files, 1):
                print(f"Processing {i}/{len(image_files)}: {image_file.name}")
                
                output_file = output_path / f"{image_file.stem}_prediction{image_file.suffix}"
                
                predictor.predict(
                    image_path=str(image_file),
                    output_path=str(output_file),
                    checkpoint_path=args.checkpoint
                )
            
            print(f"✅ All predictions saved to: {output_path}")
            
        else:
            raise NIITrainerError(f"Input path does not exist: {args.input}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return 1


def export_command(args: argparse.Namespace) -> int:
    """Handle model export command."""
    print("📦 Starting NII-Trainer Model Export")
    print("=" * 50)
    
    try:
        # Load configuration if provided
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        # Create exporter
        exporter = NIITrainer(
            config=config,
            experiment_name="cli_export"
        )
        
        # Setup dummy data path for model loading
        exporter.setup("dummy_path")
        
        # Determine input shape
        if args.input_shape:
            input_shape = tuple(args.input_shape)
        else:
            # Use default shape from config or fallback
            if hasattr(config.data, 'image_size'):
                input_shape = (1, *config.data.image_size)
            else:
                input_shape = (1, 1, 512, 512)  # Default for medical images
        
        print(f"🏋️ Model checkpoint: {args.checkpoint}")
        print(f"📁 Output path: {args.output}")
        print(f"📦 Export format: {args.format}")
        print(f"📐 Input shape: {input_shape}")
        print("=" * 50)
        
        # Load checkpoint and export
        exported_path = exporter.export_model(
            format=args.format,
            filepath=args.output,
            input_shape=input_shape
        )
        
        print(f"✅ Model exported successfully to: {exported_path}")
        
        # Show file size
        file_size = os.path.getsize(exported_path) / 1024**2  # MB
        print(f"📊 Exported model size: {file_size:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def config_command(args: argparse.Namespace) -> int:
    """Handle configuration commands."""
    if args.config_action == 'create':
        return _create_config(args)
    elif args.config_action == 'validate':
        return _validate_config(args)
    elif args.config_action == 'show':
        return _show_config(args)
    else:
        print("❌ No configuration action specified")
        return 1


def _create_config(args: argparse.Namespace) -> int:
    """Create configuration template."""
    print(f"📝 Creating {args.template} configuration template")
    print("=" * 50)
    
    try:
        if args.template == 'basic':
            config = create_default_config()
        elif args.template == 'advanced':
            config = create_default_config()
            # Customize for advanced use
            config.model.use_attention = True
            config.model.use_deep_supervision = True
            config.training.mixed_precision = True
            config.training.scheduler = "cosine"
        elif args.template == 'research':
            config = create_default_config()
            # Customize for research
            config.model.num_stages = 3
            config.model.use_attention = True
            config.model.use_auxiliary_loss = True
            config.training.training_strategy = "progressive"
            config.evaluation.compute_hausdorff = True
            config.evaluation.bootstrap_samples = 1000
        
        # Save configuration
        config.to_yaml(args.output)
        
        print(f"✅ Configuration template created: {args.output}")
        print("📝 Template type:", args.template)
        print("🔧 Edit the configuration file to customize for your use case")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        return 1


def _validate_config(args: argparse.Namespace) -> int:
    """Validate configuration file."""
    print(f"🔍 Validating configuration: {args.config}")
    print("=" * 50)
    
    try:
        config = load_config(args.config)
        config.validate()
        
        print("✅ Configuration is valid!")
        print("📊 Configuration summary:")
        print(f"  Model: {config.model.model_name}")
        print(f"  Stages: {config.model.num_stages}")
        print(f"  Training epochs: {config.training.max_epochs}")
        print(f"  Batch size: {config.data.batch_size}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def _show_config(args: argparse.Namespace) -> int:
    """Show configuration details."""
    print(f"📋 Configuration: {args.config}")
    print("=" * 50)
    
    try:
        config = load_config(args.config)
        
        # Pretty print configuration
        config_dict = config.to_dict()
        import json
        print(json.dumps(config_dict, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1


def info_command(args: argparse.Namespace) -> int:
    """Handle system information command."""
    print("ℹ️  NII-Trainer System Information")
    print("=" * 50)
    
    try:
        system_info = get_system_info()
        
        # Show general info by default or if requested
        if not any([args.system, args.gpu, args.dependencies]) or args.system:
            print("🖥️  System Information:")
            print(f"  Platform: {system_info['platform']}")
            print(f"  Python: {system_info['python_version']}")
            print(f"  PyTorch: {system_info['pytorch_version']}")
            print(f"  CPU Cores: {system_info['cpu_count']} ({system_info['cpu_count_logical']} logical)")
            print(f"  Memory: {system_info['memory_total'] / 1024**3:.1f} GB")
            print()
        
        # Show GPU info if requested or if GPUs are available
        if args.gpu or (system_info['cuda_available'] and not args.dependencies):
            print("🎮 GPU Information:")
            if system_info['cuda_available']:
                print(f"  CUDA: {system_info['cuda_version']}")
                print(f"  cuDNN: {system_info['cudnn_version']}")
                
                if 'gpu_info' in system_info:
                    gpu_info = system_info['gpu_info']
                    for i, device in enumerate(gpu_info['devices']):
                        print(f"  GPU {i}: {device['name']}")
                        print(f"    Memory: {device['total_memory'] / 1024**3:.1f} GB")
                        print(f"    Free: {device['memory_free'] / 1024**3:.1f} GB")
            else:
                print("  CUDA: Not available")
            print()
        
        # Show dependencies if requested
        if args.dependencies:
            print("📦 Dependencies:")
            try:
                import torch
                import numpy
                import nibabel
                import SimpleITK
                import sklearn
                
                print(f"  torch: {torch.__version__}")
                print(f"  numpy: {numpy.__version__}")
                print(f"  nibabel: {nibabel.__version__}")
                print(f"  SimpleITK: {SimpleITK.__version__}")
                print(f"  scikit-learn: {sklearn.__version__}")
                
                # Optional dependencies
                try:
                    import tensorboard
                    print(f"  tensorboard: {tensorboard.__version__}")
                except ImportError:
                    print("  tensorboard: Not installed")
                
                try:
                    import wandb
                    print(f"  wandb: {wandb.__version__}")
                except ImportError:
                    print("  wandb: Not installed")
                
            except ImportError as e:
                print(f"  Error checking dependencies: {e}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get system information: {e}")
        return 1