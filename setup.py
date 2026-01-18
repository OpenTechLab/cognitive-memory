"""
Cognitive Memory - Biologically Inspired Persistent Memory for LLMs

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cognitive-memory",
    version="2.0.0-beta",
    author="Michal Seidl",
    author_email="vyvoj@opentechlab.cz",
    description="Biologically inspired persistent memory system for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenTechLab/cognitive-memory",
    project_urls={
        "Bug Tracker": "https://github.com/OpenTechLab/cognitive-memory/issues",
        "Documentation": "https://github.com/OpenTechLab/cognitive-memory#readme",
        "Source": "https://github.com/OpenTechLab/cognitive-memory",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",  # CC BY-NC 4.0
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "scikit-learn>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
    },
    keywords=[
        "llm",
        "memory",
        "transformer",
        "deep-learning",
        "pytorch",
        "cognitive-science",
        "biologically-inspired",
        "persistent-memory",
        "attention-mechanism",
    ],
)
