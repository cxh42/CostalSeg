# CostalSeg: Image segmentation for costal erosion monitoring

## About
This is University of Washington ENGINE Capstone project: Machine Learning for Community-Driven Coastal Erosion Monitoring and Management.
We present a system for multi-class segmentation and analysis of community user-uploaded images, while also integrating outlier detection and multi-image perspective correction. This is an image processing system developed for coastal research at the University of Washington Applied Physics Laboratory.

[Xinghao Chen](https://cxh42.github.io/) <sup>1,</sup><sup>2</sup>, [Zheheng Li](https://github.com/Martyr12333) <sup>1,</sup><sup>2</sup>, [Dylan Scott](https://github.com/dwilsons) <sup>1,</sup><sup>2</sup>, Aaryan Shah <sup>1,</sup><sup>2</sup>, Bauka Zhandulla <sup>1,</sup><sup>2</sup>, Sarah Li <sup>1,</sup><sup>2</sup>

<sup>1 </sup>University of Washington&emsp; <sup>2 </sup>University of Washington Applied Physics Laboratory &emsp;

<div style="display: flex; justify-content: center;">
    <img src="assets/originalshow.jpg" style="width: 49%;" />
    <img src="assets/overlayshow.webp" style="width: 49%;" />
</div>

Try image segmentation demo at  

https://huggingface.co/spaces/AveMujica/MetalMarcy  
https://huggingface.co/spaces/AveMujica/SilhouetteJaenette  
https://huggingface.co/spaces/AveMujica/CostalSegment(slower, integrates outlier detection and spatial alignment, mainly used for the [MyCoast](https://mycoast.org/wa).)  

## News

## Environmental Setups
```bash
git clone https://github.com/cxh42/CostalSeg.git
cd CostalSeg
conda create -n CostalSeg python=3.12 
conda activate CostalSeg
pip install -r requirements.txt
```

## Data Preparation

**1. Training segmentation model from scratch:**

If you dont want to train model by yourself, just skip this step and turn to 2. Download our pretrained model. 

Download dataset for segmentation model training from [link]([https://drive.google.com/file/d/184yJDCdGg8OZzl6mnEC8e8TvO_cK-qFU/view?usp=sharing](https://drive.google.com/file/d/1oK9xsOT-BuuNeJHC0HfhYjtSJMIDBmPr/view?usp=sharing)), unzip the folder to project's root directory. 

Run `./SegmentModelTraining/MetalMarcy/train.py` and `./SegmentModelTraining/SilhouetteJaenette/train.py` to train model, then save your best trained .pth model to `./models`, rename them to `MM_best_model.pth` and `SJ_best_model.pth`

**2. Download our pretrained model:**

Download pretrained image segmentation model form [link](https://drive.google.com/file/d/1qGGWi3F_BLzHptIFHY33XDsABbfnalEB/view?usp=sharing), unzip the folder to project's root directory.

## Run
```bash
conda activate CostalSeg
python app.py
```
By running app.py, a graphical interactive interface will automatically open in the browser. The user interface is simple and intuitive, so there is no need to go into details about how to use it. If you need assistance, please contact [Xinghao Chen](xhc42@outlook.com).
