"""
Main CLI entry point for NII-Trainer.
"""

import argparse
import sys
from typing import List, Optional

from .commands import (
    train_command,
    evaluate_command,
    predict_command,
    export_command,
    config_command,
    info_command
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='nii-trainer',
        description='NII-Trainer: Advanced Neural Network Training Framework for Medical Image Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  nii-trainer train --config config.yaml --train-data /path/to/train --val-data /path/to/val
  
  # Quick training with default settings
  nii-trainer train --train-data /path/to/train --val-data /path/to/val --epochs 50
  
  # Evaluate a trained model
  nii-trainer evaluate --config config.yaml --test-data /path/to/test --checkpoint model.pth
  
  # Make predictions
  nii-trainer predict --input image.nii.gz --output prediction.nii.gz --checkpoint model.pth
  
  # Export model for deployment
  nii-trainer export --checkpoint model.pth --output model.onnx --format onnx
  
  # Create configuration template
  nii-trainer config create --output config.yaml
  
  # Show system information
  nii-trainer info
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='NII-Trainer 1.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train a model',
        description='Train a cascaded segmentation model'
    )
    _add_train_arguments(train_parser)
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a model',
        description='Evaluate a trained model on test data'
    )
    _add_evaluate_arguments(eval_parser)
    
    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Make predictions',
        description='Make predictions on new images'
    )
    _add_predict_arguments(predict_parser)
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export model',
        description='Export model for deployment'
    )
    _add_export_arguments(export_parser)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration utilities',
        description='Create and validate configurations'
    )
    _add_config_arguments(config_parser)
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='System information',
        description='Show system and environment information'
    )
    _add_info_arguments(info_parser)
    
    return parser


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training command arguments."""
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Training data path'
    )
    
    parser.add_argument(
        '--val-data',
        type=str,
        help='Validation data path'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Experiment name'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training'
    )


def _add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add evaluation command arguments."""
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Test data path'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['dice', 'iou'],
        help='Metrics to compute'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction images'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for evaluation'
    )


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    """Add prediction command arguments."""
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image path or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for predictions'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for prediction'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for prediction'
    )
    
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Enable test-time augmentation'
    )


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add export command arguments."""
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for exported model'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['onnx', 'torchscript'],
        default='onnx',
        help='Export format'
    )
    
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        help='Input tensor shape (e.g., 1 512 512)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add config command arguments."""
    config_subparsers = parser.add_subparsers(
        dest='config_action',
        help='Configuration actions'
    )
    
    # Create config
    create_parser = config_subparsers.add_parser(
        'create',
        help='Create configuration template'
    )
    create_parser.add_argument(
        '--output', '-o',
        type=str,
        default='config.yaml',
        help='Output configuration file path'
    )
    create_parser.add_argument(
        '--template',
        type=str,
        choices=['basic', 'advanced', 'research'],
        default='basic',
        help='Configuration template type'
    )
    
    # Validate config
    validate_parser = config_subparsers.add_parser(
        'validate',
        help='Validate configuration file'
    )
    validate_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration file to validate'
    )
    
    # Show config
    show_parser = config_subparsers.add_parser(
        'show',
        help='Show configuration'
    )
    show_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration file to show'
    )


def _add_info_arguments(parser: argparse.ArgumentParser) -> None:
    """Add info command arguments."""
    parser.add_argument(
        '--system',
        action='store_true',
        help='Show system information'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Show GPU information'
    )
    
    parser.add_argument(
        '--dependencies',
        action='store_true',
        help='Show dependency versions'
    )


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set up logging based on verbosity
    import logging
    if parsed_args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Dispatch to appropriate command
    try:
        if parsed_args.command == 'train':
            return train_command(parsed_args)
        elif parsed_args.command == 'evaluate':
            return evaluate_command(parsed_args)
        elif parsed_args.command == 'predict':
            return predict_command(parsed_args)
        elif parsed_args.command == 'export':
            return export_command(parsed_args)
        elif parsed_args.command == 'config':
            return config_command(parsed_args)
        elif parsed_args.command == 'info':
            return info_command(parsed_args)
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())