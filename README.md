# ExamHandOCR: Official Implementation

This repository contains the official implementation of the paper **"ExamHandOCR: A Benchmark Dataset for Long-Form Handwritten Answer Sheet OCR in Examination Scenarios"** (NeurIPS 2026 Datasets and Benchmarks Track).

## Overview

ExamHandOCR is the largest and most operationally grounded benchmark dataset for long-form handwritten answer-sheet OCR, comprising **3,158,804 images** from real-world examination pipelines.

### Key Features

- **3.16M images** from live online marking systems (网上阅卷)
- **8,640 professionally annotated** images with 99.2% IAA
- **Three novel benchmark tracks**:
  1. Semi-Supervised Long-Form OCR (SSL-OCR)
  2. Cross-Session Generalization (CSG)
  3. Operational-Fidelity Evaluation (OFE)
- **Novel evaluation metric**: ExamScore-Aware CER (ESA-CER)

## Installation

```bash
# Clone the repository
git clone https://github.com/examhandocr/ExamHandOCR.git
cd ExamHandOCR/CODE

# Create conda environment
conda create -n examhandocr python=3.9
conda activate examhandocr

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU training)
- See `requirements.txt` for full dependency list

## Quick Start

### 1. Data Preparation

Download the dataset from [Zenodo](https://zenodo.org/records/19145349) and organize as follows:

```
data/
├── Current/              # Standard pipeline branch
│   ├── 2024GKZH001_MATH/
│   │   ├── <PSID>/
│   │   │   ├── 01.jpg
│   │   │   └── ...
│   └── ...
├── Current_jst/          # Alternative pipeline branch
├── annotations.json      # 8,640 annotated samples
└── splits.json          # Train/val/test splits
```

### 2. Training

#### Train OCR Model (e.g., TrOCR)

```bash
python main.py train \
    --model trocr \
    --data_root ./data/ExamHandOCR \
    --annotation_file ./data/annotations.json \
    --output_dir ./outputs/trocr \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --device cuda
```

#### SSL Pre-training (MAE)

```bash
python main.py pretrain \
    --data_root ./data/ExamHandOCR/train-unsup \
    --output_dir ./outputs_ssl \
    --batch_size 256 \
    --epochs 100 \
    --lr 1.5e-4 \
    --mask_ratio 0.75 \
    --device cuda
```

#### Train with SSL Pre-trained Encoder

```bash
python main.py train \
    --model trocr \
    --ssl_pretrained \
    --ssl_path ./outputs_ssl/best_mae.pth \
    --data_root ./data/ExamHandOCR \
    --annotation_file ./data/annotations.json \
    --output_dir ./outputs/trocr_ssl
```

### 3. Evaluation

#### Standard Evaluation

```bash
python main.py evaluate \
    --model_path ./outputs/trocr/best_model.pth \
    --model_type trocr \
    --data_root ./data/ExamHandOCR \
    --annotation_file ./data/annotations.json \
    --output_file ./results/trocr_results.json \
    --calculate_oqs
```

#### Evaluate Benchmark Tracks

```bash
# Operational-Fidelity Evaluation (OFE)
python main.py evaluate_track \
    --track operational_fidelity \
    --model_path ./outputs/trocr/best_model.pth \
    --model_type trocr \
    --data_root ./data/ExamHandOCR \
    --output_file ./results/ofe_results.json
```

## Project Structure

```
CODE/
├── data/                   # Data loading and preprocessing
│   ├── dataset.py         # Dataset classes
│   ├── transforms.py      # Data augmentation
│   ├── tokenizer.py       # Text tokenization
│   └── dataloader.py      # DataLoader utilities
├── models/                 # Model implementations
│   ├── crnn.py            # CRNN baseline
│   ├── abinet.py          # ABINet
│   ├── trocr.py           # TrOCR + SSL variant
│   ├── vit_ocr.py         # ViT-OCR
│   ├── layout_models.py   # Layout analysis models
│   └── ssl_mae.py         # MAE for SSL pre-training
├── metrics/                # Evaluation metrics
│   ├── cer_wer.py         # CER/WER
│   ├── esa_cer.py         # ExamScore-Aware CER
│   ├── oqs.py             # Operational Quality Score
│   ├── ri.py              # Robustness Index
│   └── layout_metrics.py  # Layout evaluation
├── train/                  # Training scripts
│   ├── train_ocr.py       # OCR model training
│   ├── train_ssl.py       # SSL pre-training
│   └── train_layout.py    # Layout model training
├── eval/                   # Evaluation scripts
│   ├── evaluate_ocr.py    # Standard evaluation
│   ├── evaluate_layout.py # Layout evaluation
│   └── evaluate_tracks.py # Track-specific evaluation
├── utils/                  # Utilities
│   ├── checkpoint.py      # Checkpoint management
│   ├── logger.py          # Logging
│   ├── config.py          # Configuration
│   └── visualization.py   # Visualization
├── configs/                # Configuration files
│   ├── default.yaml       # Default config
│   └── ssl_pretrain.yaml  # SSL pre-training config
├── main.py                 # Main entry point
└── README.md               # This file
```

## Implemented Models

### OCR Models

| Model | Paper | CER (Paper) | ESA-CER (Paper) |
|-------|-------|-------------|-----------------|
| CRNN | Shi et al. 2016 | 18.72% | 24.31% |
| ABINet | Fang et al. 2021 | 12.34% | 16.42% |
| TrOCR-Base | Li et al. 2023 | 8.71% | 11.83% |
| ViT-OCR | Diaz et al. 2021 | 7.23% | 9.74% |
| **TrOCR + SSL** | **Ours** | **5.84%** | **7.62%** |
| Human | - | 0.80% | 1.12% |

### Layout Models

| Model | mIoU | F1@0.5 | F1@0.75 |
|-------|------|--------|---------|
| U-Net | 78.34% | 82.17% | 64.31% |
| Mask R-CNN | 83.91% | 87.43% | 71.88% |
| DETR | 85.72% | 89.21% | 74.53% |
| **DETR + SSL** | **88.14%** | **91.63%** | **78.42%** |

## Benchmark Tracks

### Track 1: Semi-Supervised Long-Form OCR (SSL-OCR)

- **Protocol**: Pre-train on 3.15M unannotated images with MAE, fine-tune on 6,048 annotated images
- **Key Metric**: CER at varying annotation set sizes (100 → 6,048)
- **Result**: SSL pre-training achieves **14× reduction** in annotation requirement

### Track 2: Cross-Session Generalization (CSG)

- **Protocol**: Train on Current tree, evaluate zero-shot on Current_jst tree
- **Key Metric**: Generalization Gap (GG) = |CER_source - CER_target|
- **Result**: Zero-shot GG ≈ 7.6-8.7%, reduced to 1.7-2.1% with 5% target adaptation

### Track 3: Operational-Fidelity Evaluation (OFE)

- **Protocol**: Stratify test set by OQS into High/Medium/Low quality tiers
- **Key Metrics**: CER per tier, Robustness Index (RI)
- **Result**: RI = 0.69 for TrOCR + SSL (closer to 1.0 = more robust)

## Evaluation Metrics

### Standard Metrics
- **CER**: Character Error Rate
- **WER**: Word Error Rate

### Novel Metrics
- **ESA-CER**: ExamScore-Aware CER (α=3.0 for math tokens)
  ```
  ESA-CER = Σ(w_i × 1[char_i is erroneous]) / Σ(w_i)
  ```
- **OQS**: Operational Quality Score (composite of skew, contrast, JPEG blocking, bleed-through)
- **RI**: Robustness Index (sensitivity to image quality)
- **GG**: Generalization Gap (cross-session transfer)

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@inproceedings{examhandocr2026,
  title={ExamHandOCR: A Benchmark Dataset for Long-Form Handwritten Answer Sheet OCR in Examination Scenarios},
  author={[Authors]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2026}
}
```

## License

This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (Grant No. 62306187, 62376144, 62332016)
- Beijing Municipal Science & Technology Commission (Z231100007923016)
- Shanghai AI Laboratory Open Research Fund
- Tsinghua University Initiative Scientific Research Program (2023Z08QCX01)

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors at [email]

---

**Note**: This is a research dataset intended for academic use. Please refer to the Data Use Agreement for terms and conditions.
