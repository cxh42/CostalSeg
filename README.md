# CostalSeg: Image segmentation for costal erosion monitoring

## About
University of Washington Capstone project: Machine Learning for Community-Driven Coastal Erosion Monitoring and Management.  
We present a system for multi-class segmentation and analysis of community user-uploaded images, while also integrating outlier detection and multi-image perspective correction.  This is an image processing system developed for coastal research at the University of Washington Applied Physics Laboratory.  
Segmentation Model: DeepLabV3Plus with EfficientNet-B6, achieved 0.93 IoU score.

[Xinghao Chen](https://cxh42.github.io/) <sup>1,</sup><sup>2</sup>, [Zheheng Li](https://github.com/Martyr12333) <sup>1,</sup><sup>2</sup>, [Dylan Scott](https://github.com/dwilsons) <sup>1,</sup><sup>2</sup>, Aaryan Shah <sup>1,</sup><sup>2</sup>, Bauka Zhandulla <sup>1,</sup><sup>2</sup>, Sarah Li <sup>1,</sup><sup>2</sup>

<sup>1 </sup>University of Washington&emsp; <sup>2 </sup>University of Washington Applied Physics Laboratory &emsp;

<div style="display: flex; justify-content: center;">
    <img src="assets/originalshow.jpg" style="width: 49%;" />
    <img src="assets/overlayshow.webp" style="width: 49%;" />
</div>

Try image segmentation demo at  

https://huggingface.co/spaces/AveMujica/MetalMarcy  
https://huggingface.co/spaces/AveMujica/SilhouetteJaenette  
https://huggingface.co/spaces/AveMujica/CostalSegment (slower, integrates outlier detection and spatial alignment, mainly used for the [MyCoast](https://mycoast.org/wa).)  

## Environmental Setups
```bash
git clone https://github.com/cxh42/CostalSeg.git
cd CostalSeg
conda create -n CostalSeg python=3.12
conda activate CostalSeg
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data Preparation

**1. One-click download of models and datasets (recommended):**

Run a single script to fetch everything to the right places:

```bash
pip install -r requirements.txt
python scripts/fetch_assets.py
```

What it does

- Downloads pretrained models to `models/`:
  - `MM_best_model.pth` from https://huggingface.co/AveMujica/CostalSeg-MM
  - `SJ_best_model.pth` from https://huggingface.co/AveMujica/CostalSeg-SJ
- Syncs datasets from Hugging Face Datasets to training folders:
  - `AveMujica/CostalSeg-MM` -> `SegmentModelTraining/MetalMarcy/dataset`
  - `AveMujica/CostalSeg-SJ` -> `SegmentModelTraining/SilhouetteJaenette/dataset`
- Optionally downloads reference vectors if missing (`models/*_mean.npy`).

You can force re-download by adding `--force`.

**2. Training segmentation model from scratch:**

If you dont want to train model by yourself, just use 1. One-click setup above. 

Datasets are NOT bundled in this repository. Please download them from Hugging Face (see section 3 below). After download, the folder layout expected by `train.py` is:

- `SegmentModelTraining/<Site>/dataset/train/*.jpg` with matching `*_mask.png`
- `SegmentModelTraining/<Site>/dataset/valid/*.jpg` with matching `*_mask.png`
- `SegmentModelTraining/<Site>/dataset/test/*.jpg` with matching `*_mask.png`

Run `./SegmentModelTraining/MetalMarcy/train.py` and `./SegmentModelTraining/SilhouetteJaenette/train.py` to train model, then save your best trained .pth model to `./models`, rename them to `MM_best_model.pth` and `SJ_best_model.pth`

Start training (two sites):

```bash
# Metal Marcy
python SegmentModelTraining/MetalMarcy/train.py

# Silhouette Jaenette
python SegmentModelTraining/SilhouetteJaenette/train.py
```

Notes

- The scripts expect masks to be named as `<image_basename>_mask.png` in the same folder as the images.
- Uses PyTorch Lightning; GPU is auto-detected if available. On Windows, `num_workers=0` is already set.

**3. Pretrained models**

No manual download needed. Running `python scripts/fetch_assets.py` will place the weights at:

- `models/MM_best_model.pth`
- `models/SJ_best_model.pth`

For reference, models are hosted on:

- https://huggingface.co/AveMujica/CostalSeg-MM
- https://huggingface.co/AveMujica/CostalSeg-SJ

Optional: reference vectors for outlier detection

`app.py` also uses `models/MM_mean.npy` and `models/SJ_mean.npy` if present; missing vectors do not block running the app.



## Run
```bash
conda activate CostalSeg
python app.py
```
By running app.py, a graphical interactive interface will automatically open in the browser. The user interface is simple and intuitive, so there is no need to go into details about how to use it. If you need assistance, please contact Xinghao Chen xhc42@outlook.com

### Built-in Examples

The UI includes example images you can load with one click (see the "Examples" blocks in both tabs):

- Single Image Segmentation: preloaded samples from `reference_images/MM` and `reference_images/SJ`.
- Spatial Alignment Segmentation: paired examples (reference + target) for both sites.

You can also drop your own images into the inputs; no special formatting required.

## Batch Processing (CLI)

For bulk processing without GUI, use the CLI script `batch_infer.py`. It supports two workflows:

- Segmentation on a folder of images
- Spatial alignment (one reference) + segmentation for a folder of targets

Ensure pretrained weights are in place (see above) before running.

Examples

```bash
# 1) Batch segmentation (save segmentation map + overlay)
python batch_infer.py segment \
  --location "Metal Marcy" \
  --input path/to/images \
  --output outputs/mm_segment \
  --overlay

# 2) Batch segmentation with outlier detection (adds outlier column to summary.csv)
python batch_infer.py segment \
  --location "Silhouette Jaenette" \
  --input path/to/images \
  --output outputs/sj_segment \
  --outlier

# 3) Batch spatial alignment + segmentation
python batch_infer.py align \
  --location "Metal Marcy" \
  --reference reference_images/MM/2025-01-26_16-36-00_MM.jpg \
  --input path/to/targets \
  --output outputs/mm_aligned \
  --overlay
```

Outputs

- For each input image: `*_seg.png` (segmentation map), `*_overlay.png` (if `--overlay`).
- Per-run CSV summary: `summary.csv` containing percentages of each class and optional `outlier` column.
