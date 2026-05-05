# ExamHandOCR: A Benchmark Dataset for Long-Form Handwritten Answer Sheet OCR in Examination Scenarios

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the implementation of the paper **"ExamHandOCR: A Benchmark Dataset for Long-Form Handwritten Answer Sheet OCR in Examination Scenarios"** submitted to NeurIPS 2026 Datasets and Benchmarks Track.

---

## Overview

ExamHandOCR is a benchmark dataset for long-form handwritten answer-sheet OCR, comprising **3,158,804 images** from real-world examination marking pipelines.

### Key Statistics

| Property | Value |
|----------|-------|
| Total Images | 3,158,804 |
| Annotated Images | 8,640 |
| Unannotated Images | 3,150,164 |
| Examination Sessions | 1,247 |
| Pseudonymous Examinees | 312,451 |
| Subjects | 8 (Chinese, Math, English, Physics, Chemistry, History, Geography, Biology) |
| Time Span | 2019-2024 |

### Benchmark Tracks

1. **Semi-Supervised Long-Form OCR (SSL-OCR)**
2. **Cross-Session Generalization (CSG)**
3. **Operational-Fidelity Evaluation (OFE)**

---

## Installation

```bash
# Clone the repository
git clone https://github.com/FrankDengAI/ExamHandOCR.git
cd ExamHandOCR

# Create conda environment
conda create -n examhandocr python=3.9
conda activate examhandocr

# Install dependencies
pip install -r CODE/requirements.txt
```

---

## Quick Start

### 1. Train OCR Model

```bash
python CODE/main.py train \
    --model trocr \
    --data_root ./data \
    --annotation_file ./data/annotations.json \
    --output_dir ./outputs/trocr \
    --batch_size 32 \
    --epochs 50
```

### 2. SSL Pre-training

```bash
python CODE/main.py pretrain \
    --data_root ./data/train-unsup \
    --output_dir ./outputs_ssl \
    --batch_size 256 \
    --epochs 100
```

### 3. Evaluate

```bash
python CODE/main.py evaluate \
    --model_path ./outputs/trocr/best_model.pth \
    --model_type trocr \
    --data_root ./data \
    --annotation_file ./data/annotations.json
```

---

## Benchmark Results

### OCR Models

| Model | CER (%) | WER (%) | ESA-CER (%) | RI |
|-------|---------|---------|-------------|-----|
| CRNN | 18.72 | 27.43 | 24.31 | 0.34 |
| ABINet | 12.34 | 19.87 | 16.42 | 0.41 |
| TrOCR-Base | 8.71 | 14.26 | 11.83 | 0.53 |
| ViT-OCR | 7.23 | 12.08 | 9.74 | 0.61 |
| **TrOCR + SSL** | **5.84** | **9.71** | **7.62** | **0.69** |

---

## Repository Structure

```
ExamHandOCR/
├── CODE/                       # Implementation
│   ├── data/                  # Data loading
│   ├── models/                # Model implementations
│   ├── metrics/               # Evaluation metrics
│   ├── train/                 # Training scripts
│   ├── eval/                  # Evaluation scripts
│   └── main.py                # Entry point
├── examhandocr_croissant.json # Dataset metadata
└── README.md                  # This file
```

---

## License

This project is licensed under the CC BY-NC 4.0 License.

---

*This is an anonymized version of the repository for peer review.*
