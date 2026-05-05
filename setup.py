#!/usr/bin/env python3
"""
Setup script for ExamHandOCR package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="examhandocr",
    version="1.0.0",
    author="ExamHandOCR Team",
    author_email="examhandocr@example.com",
    description="ExamHandOCR: A Benchmark Dataset for Long-Form Handwritten Answer Sheet OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/examhandocr/ExamHandOCR",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "examhandocr=main:main",
        ],
    },
)
