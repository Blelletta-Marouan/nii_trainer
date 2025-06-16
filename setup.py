"""
NII-Trainer: Advanced Neural Network Training Framework for Medical Image Segmentation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nii-trainer",
    version="1.0.0",
    author="NII-Trainer Development Team",
    author_email="dev@nii-trainer.org",
    description="Advanced neural network training framework for medical image segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/nii-trainer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.10",
        ],
        "optuna": ["optuna>=2.10"],
        "wandb": ["wandb>=0.12"],
        "tensorboard": ["tensorboard>=2.8"],
        "onnx": ["onnx>=1.10", "onnxruntime>=1.10"],
    },
    entry_points={
        "console_scripts": [
            "nii-trainer=nii_trainer.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nii_trainer": ["configs/*.yaml", "templates/*.yaml"],
    },
    keywords=[
        "deep learning",
        "medical imaging",
        "segmentation",
        "neural networks",
        "pytorch",
        "medical ai",
        "computer vision",
        "healthcare",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/nii-trainer/issues",
        "Source": "https://github.com/your-org/nii-trainer",
        "Documentation": "https://nii-trainer.readthedocs.io/",
    },
)