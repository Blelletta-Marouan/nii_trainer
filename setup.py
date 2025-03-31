from setuptools import setup, find_packages

setup(
    name="nii_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "nibabel>=3.2.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A flexible framework for training neural networks on NIfTI data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nii_trainer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.7"
)