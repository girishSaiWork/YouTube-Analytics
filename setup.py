#!/usr/bin/env python3
"""
Setup script for YouTube Analytics project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="youtube-analytics",
    version="0.1.0",
    description="A PySpark-based data engineering project for analyzing YouTube trending videos data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/youtube-analytics",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "youtube-analytics=scripts.run_pipeline:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    
    # Keywords
    keywords="youtube analytics pyspark data-engineering spark",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/youtube-analytics/issues",
        "Source": "https://github.com/yourusername/youtube-analytics",
        "Documentation": "https://github.com/yourusername/youtube-analytics/wiki",
    },
)
