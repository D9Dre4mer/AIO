"""
Setup script for Advanced LightGBM Optimization Project
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-lightgbm-optimization",
    version="1.0.0",
    author="Advanced ML Team",
    author_email="team@advancedml.com",
    description="Advanced LightGBM optimization with cutting-edge techniques",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/advancedml/advanced-lightgbm-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "cupy>=12.0.0",
            "nvidia-cudnn-cu12>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "advanced-lightgbm=main:main",
            "lightgbm-optimize=run_optimization:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    zip_safe=False,
)
