# CostalSeg: Image segmentation for costal erosion monitoring
# Note: this project is working on progress!!!

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

**For segmentation model training datasets from scratch:**

**Download our pretrained model:**
Down load pre trained image segmentation form [link](https://drive.google.com/file/d/1qGGWi3F_BLzHptIFHY33XDsABbfnalEB/view?usp=sharing), unzip the folder to project's root directory.

## Run
```bash
conda activate CostalSeg
python app.py
```
Then open the link to use visualized pannel.

## Contributions

## Acknowledgement

## Citation