# CostalSeg — Installation & User Guide (Windows + NVIDIA GPU)

**Project URL:** https://github.com/cxh42/CostalSeg

**Overview**  
CostalSeg is a project for coastal image segmentation. This guide explains how to set up a Windows + NVIDIA GPU environment, fetch pretrained assets, (optionally) retrain, and launch the app for interactive use.

---

## 1) Prerequisites

- **Operating system:** Windows 10/11 (64-bit)
- **Hardware:** NVIDIA GPU (recent driver recommended for CUDA 12.x runtime wheels)
- **Disk space:** ~1.31 GB
- **Recommended tools:** Miniconda (or Anaconda), VS Code, Git

> Tip: You can verify your GPU/driver with `nvidia-smi` in a Command Prompt (or PowerShell).

---

## 2) Clone the repository

```bash
git clone https://github.com/cxh42/CostalSeg.git
cd CostalSeg
```

---

## 3) Create and activate a Conda environment (Python 3.12)

```bash
conda create -n CostalSeg python=3.12
conda activate CostalSeg
```

---

## 4) Install PyTorch (CUDA 12.6 build) and project dependencies

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**(Optional) Quick GPU check in Python:**
```python
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5) One-click download of pretrained models and datasets

```bash
python scripts/fetch_assets.py
```
This script downloads the **trained models** and the **datasets used in training** into the correct locations for immediate use.

---

## 6) (Optional) Retrain the models locally

If you want to reproduce training locally:
```bash
# Train the "Metal Marcy" model
python SegmentModelTraining/MetalMarcy/train.py

# Train the "Silhouette Jaenette" model
python SegmentModelTraining/SilhouetteJaenette/train.py
```
These scripts will run end-to-end and place the resulting model weights where the app can find them automatically. If you only want to use pretrained models, you can skip this section.

---

## 7) Launch the interactive app (GUI)

```bash
python app.py
```
Running this command opens a browser-based interface. You can upload your own images for processing, and sample images are included in the repository. If a browser tab does not open automatically, copy the local URL printed in the terminal into your browser.

---

## 8) Troubleshooting

- **GPU not detected (`torch.cuda.is_available() == False`)**  
  - Ensure the correct PyTorch wheel is installed (CUDA 12.6 build as shown above).  
  - Update to a recent NVIDIA driver compatible with CUDA 12.x.  
  - Close and reopen your terminal, then re-activate the Conda environment.

- **Browser doesn’t open automatically**  
  - Press Enter again

---

## 9) License

See the repository’s `LICENSE` file for the project’s license details.

---

## 10) Quick command summary (copy-paste)

```bash
# Clone
git clone https://github.com/cxh42/CostalSeg.git
cd CostalSeg

# Conda env
conda create -n CostalSeg python=3.12
conda activate CostalSeg

# Install
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Fetch assets
python scripts/fetch_assets.py

# (Optional) Train
python SegmentModelTraining/MetalMarcy/train.py
python SegmentModelTraining/SilhouetteJaenette/train.py

# Launch app
python app.py
```
