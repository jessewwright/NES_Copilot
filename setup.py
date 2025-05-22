"""
Setup script for NES Copilot

This script installs the NES Copilot package.
"""

from setuptools import setup, find_packages

setup(
    name="nes_copilot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "torch>=1.13.0",  # More flexible PyTorch version
        "sbi==0.22.0",
        "transformers>=4.18.0",
        "joblib>=1.1.0",
        "pebble>=4.6.0"
    ],
    author="NES Copilot Team",
    author_email="example@example.com",
    description="An agentic system for automating cognitive model development and validation",
    keywords="cognitive modeling, simulation-based inference, neural posterior estimation",
    python_requires=">=3.8",
)
