# Wildfire Detection: Dual-Stream CNN on the FLAME Dataset

This repository implements and benchmarks 8 Convolutional Neural Network (CNN) architectures on the **FLAME (Fire Luminosity Airborne-sensing Dataset)** for early-stage forest fire detection using aerial RGB and Thermal Infrared (IR) imagery.

## Project Overview

Early detection of forest fires is critical to mitigating ecological and human losses. While standard RGB sensors can detect smoke and visible flames under clear daylight, they fail under dense smoke or nighttime conditions. Infrared (IR) sensors capture thermal signatures directly, but lack spatial context.

This project implements a **Dual-Stream CNN Fusion** architecture:
- **RGB Stream**: Processes visible light spectrum (3 channels) for color and textural context.
- **Thermal IR Stream**: Processes the thermal images, which FLAME ships as 3-channel pseudo-color, for heat signatures.
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

### Recorded Results

The training logs in [`eval.ipynb`](eval.ipynb) cover 5 runs (index 0 to 4) per model and input mode. Each run used `--subset_rate 0.01`, so **1% of FLAME**, split 80/20, which leaves about 107 test images per run. The figure below is the best test accuracy reached across epochs, averaged over the 5 runs:

| Model                   | RGB    | IR     | Both (fusion) |
| ----------------------- | ------ | ------ | ------------- |
| ResNet-18               | 99.81% | 97.57% | 99.26%        |
| VGG-16                  | 99.07% | 98.88% |               |
| MobileNet-V2            | 99.07% | 99.44% | 99.26%        |
| Flame CNN (one-stream)  | 97.76% | 90.84% | 97.38%        |
| Flame CNN (two-stream)  |        |        | 98.88%        |
| LeNet-5 (one-stream)    | 86.55% | 93.83% | 96.07%        |
| LeNet-5 (two-stream)    |        |        | 89.16%        |
| Logistic (two-stream)   | 93.64% | 94.58% | 96.26%        |

How to read this:

- **These numbers are optimistic.** Taking the best epoch by test accuracy selects the model on the test set itself, and with about 107 test images one image is worth 0.93 points. A held-out validation split and the full dataset would give a fairer estimate.
- **Fusion does not win across the board.** It helps the weaker models (LeNet-5, the logistic baseline, the Flame CNN), but RGB alone matches or beats it for the pretrained backbones.
- F1, precision and recall are computed by `train.py` into `log/results.csv` at the end of each run, but no results file from these runs was kept, so they are not reported here.

---

## Project Structure

```text
├── dataset.py         # Custom PyTorch Dataset for loading RGB + IR image pairs
├── models.py          # PyTorch model definitions (ResNet, VGG, MobileNet, Flame CNNs)
├── train.py           # Training and validation pipeline with CLI arguments
├── eval.ipynb         # Evaluation notebook with confusion matrices and performance metrics
├── requirements.txt   # Python dependency list
├── metrics.py         # F1, precision and recall for the results log
├── tests/             # Offline forward-pass and metrics tests
├── log/               # train.py writes results.csv and per-run .npy logs here
└── README.md          # Project documentation
```

---

## Setup & Installation

### Requirements
Ensure you have Python 3.10+ and a CUDA-capable GPU (highly recommended). `requirements.txt` installs CPU wheels from PyPI; for CUDA, install torch and torchvision first by following https://pytorch.org/get-started/locally/.

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

---

## Running the tests

```bash
pip install -r requirements.txt pytest
pytest -q tests
```

The tests build every architecture with pretrained downloads disabled, run a forward pass in each input mode, check the metrics helper, and import `train.py` so dependency API breaks are caught. They need no dataset or GPU.
