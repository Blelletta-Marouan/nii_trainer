"""
Test suite for NII-Trainer.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Test fixtures and utilities
from .fixtures import (
    sample_config,
    sample_image,
    sample_label,
    temp_data_dir,
    mock_model
)

# Test modules
from .test_core import *
from .test_models import *
from .test_data import *
from .test_training import *
from .test_evaluation import *
from .test_api import *
from .test_cli import *
from .test_utils import *

# Integration tests
from .integration import *

__all__ = [
    "sample_config",
    "sample_image", 
    "sample_label",
    "temp_data_dir",
    "mock_model"
]