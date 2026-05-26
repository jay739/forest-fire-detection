# Wildfire Detection — Dual-Stream CNN on FLAME Dataset

This repository implements and benchmarks 8 Convolutional Neural Network (CNN) architectures on the **FLAME (Fire Luminosity Airborne-sensing Dataset)** for early-stage forest fire detection using aerial RGB and Thermal Infrared (IR) imagery.

## Project Overview

Early detection of forest fires is critical to mitigating ecological and human losses. While standard RGB sensors can detect smoke and visible flames under clear daylight, they fail under dense smoke or nighttime conditions. Infrared (IR) sensors capture thermal signatures directly, but lack spatial context.

This project implements a **Dual-Stream CNN Fusion** architecture:
- **RGB Stream**: Processes visible light spectrum (3 channels) for color and textural context.
- **Thermal IR Stream**: Processes thermal signatures (1 channel) for heat signatures.
- **Fusion Layer**: Concatenates feature representations to perform 3-class classification (Fire, No Fire, Smoke).

---

## Model Architectures & Benchmarks

We compare single-stream (RGB or IR only) vs. dual-stream fusion across 8 architectures:
1. **Flame CNN** (One-stream & Two-stream custom architectures)
2. **ResNet-18** (One-stream & Two-stream)
3. **VGG-16** (One-stream & Two-stream)
4. **MobileNet-V2** (One-stream & Two-stream)
5. **LeNet-5** (One-stream & Two-stream)
6. **Logistic Regression Baselines**

### Key Results
* **Best Performing Model**: Dual-Stream ResNet-18.
* **Accuracy/F1 Metric**: Achieved **0.94 micro-F1** score on test partition.
* **Key Finding**: Late feature fusion of RGB and IR networks reduces false positives by 42% compared to single-stream RGB networks.

---

## Project Structure

```text
├── dataset.py         # Custom PyTorch Dataset for loading RGB + IR image pairs
├── models.py          # PyTorch model definitions (ResNet, VGG, MobileNet, Flame CNNs)
├── train.py           # Training and validation pipeline with CLI arguments
├── eval.ipynb         # Evaluation notebook with confusion matrices and performance metrics
├── requirements.txt   # Python dependency list
├── log/
│   └── results.csv    # Logging output directory for tracking metrics
└── README.md          # Project documentation
```

---

## Setup & Installation

### Requirements
Ensure you have Python 3.8+ and a CUDA-capable GPU (highly recommended).

```bash
pip install -r requirements.txt
```

### Dataset Preparation
Download the FLAME dataset and structure your directories as:
```text
/data/FLAME/
    ├── 254p RGB Images/
    └── 254p Thermal Images/
```

---

## Usage

### Training a Model
To train the one-stream custom Flame CNN on RGB images:
```bash
python train.py --model Flame_one_stream --mode rgb --path_rgb "/data/FLAME/254p RGB Images" --EPOCH 50 --batch_size 64
```

To train the dual-stream ResNet-18:
```bash
python train.py --model Resnet18_two_stream --mode both --path_rgb "/data/FLAME/254p RGB Images" --path_ir "/data/FLAME/254p Thermal Images" --EPOCH 50 --batch_size 32 --lr 0.0001
```

### Options
- `--model`: `Flame_one_stream`, `Flame_two_stream`, `VGG16`, `Vgg_two_stream`, `Resnet18`, `Resnet18_two_stream`, `Mobilenetv2`, `Mobilenetv2_two_stream`, etc.
- `--mode`: `rgb`, `ir`, or `both` (for dual-stream).
- `--EPOCH`: Number of training loops (default: `50`).
- `--lr`: Learning rate (default: `1e-3`).
