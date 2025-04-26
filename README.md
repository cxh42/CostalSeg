# Note: this project is working on progress!!!
# CostalSeg: Image segmentation for costal erosion monitoring


University of Washington ENGINE Capstone project: Machine Learning for Community-Driven Coastal Erosion Monitoring and Management

[Xinghao Chen](https://cxh42.github.io/) <sup>1*</sup>, []

<sup>1 </sup>University of Washington&emsp;

![block](assets/originalshow.jpg)![block](assets/overlayshow.webp)

Try image segmentation demo at

https://huggingface.co/spaces/AveMujica/MetalMarcy

https://huggingface.co/spaces/AveMujica/SilhouetteJaenette

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

**Training segmentation model from scratch:**

**If you dont want to train model by yourself, just skip this step.** Download dataset for segmentation model training[link](https://drive.google.com/file/d/184yJDCdGg8OZzl6mnEC8e8TvO_cK-qFU/view?usp=sharing), unzip the folder to project's roo directory. Run ./SegmenModelTraining/MetalMarcy/train.py and ./SegmenModelTraining/SilhouetteJaenette/train.py to train model, then save your best trained .pth model to ./models, rename them to MM_best_model.pth and SJ_best_model.pth

**Download our pretrained model:**

Download pretrained image segmentation model form [link](https://drive.google.com/file/d/1qGGWi3F_BLzHptIFHY33XDsABbfnalEB/view?usp=sharing), unzip the folder to project's root directory.

## Run
```bash
conda activate CostalSeg
python app.py
```
Then open the link to use visualized pannel.

## Contributions

## Acknowledgement

## Citation