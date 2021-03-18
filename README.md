# RANZCR-CLiP 7th Place Solution

This repository is **WIP**. (18 Mar 2021)

![pipeline](https://pbs.twimg.com/media/EwqG7gdVcAEOmHS?format=jpg&name=large)

## Installation
```bash
git clone https://github.com/analokmaus/kaggle-ranzcr-clip-public.git 
cd kaggle-ranzcr-clip-public
git clone https://github.com/analokmaus/kuma_utils.git 
```
`kuma_utils` is a toolbox I use for competitions and work. [Check it out!](https://github.com/analokmaus/kuma_utils)  

### conda
```bash
conda env create -n {NEW NAME} -f environment.yaml
```

### docker
WIP


**IMPORTANT: timm version**  
Since segmentation_models_pytorch requires timm=0.3.2 which does not include ResNet200D.  
I added latest timm=0.3.4 as `timm_latest` in the root directory.  
In case you need ResNet200D, you must use `import timm_latest`.


## Download datasets
```
┣ data
┃   ┣ ranzcr-clip
┃       ┣ (competition files)
┃       ┣ nih_chestxray
┃       ┃   ┣ (nih dataset)
┃       ┣ padchest
┃       ┃   ┣ (padchest dataset)
┃       ┣ mimic
┃           ┣ (mimic dataset)
┃
┣ kaggle-ranzcr-clip-public
    ┣ scripts
```
### competition files
```bash
kaggle competitions download ranzcr-clip-catheter-line-classification
```
### nih dataset
```bash
kaggle datasets download nih-chest-xrays/data
```
### padchest dataset
```bash
kaggle datasets download raddar/padchest-tubes
```
### mimic dataset
Due to the license, we cannot host MIMIC CXR dataset.  
Please go to [MIMIC CXR official website](https://mimic-cxr.mit.edu) and download by yourself.


## Benchmark
### UNet-CNN (R1)
CV: 0.9661  
Public LB: 0.970  
Private LB: 0.973  
```bash
python train.py --config Segmentation13
python train.py --config SegAndCls12
python inference.py --config SegAndCls12 # generate pseudo labels
python train_external.py --config PretrainStudent08l
python train.py --config SegAndCls12external6
```

### UNet-CNN (E1)
CV: 0.9660  
Public LB: 0.972  
Private LB: 0.973  
```bash
python train.py --config Segmentation15
python train.py --config SegAndCls14
python inference.py --config SegAndCls14 # generate pseudo labels
python train_external.py --config PretrainStudent09
python train.py --config SegAndCls14external2
```

### Vanilla CNN (N1)
CV: 0.9671  
Public LB: 0.970  
Private LB: 0.972  
```bash
(run training script by Y.Nakama)
python inference.py --config SingleModel02 # generate pseudo labels
python train_external.py --config Distillation03
python train.py --config SingleModel02external0
```

## Test Environment
Adjust batch_size and relevant parameters (learning rate etc.) when you run script.

A machine with four V100 16GB (64GB total) was used to train the following configs:
- Segmentation13
- Segmentation15
- SegAndCls12*
- SegAndCls14*
- PretrainStudent08*
- PretrainStudent09*

A machine with two GF RTX 3090 24GB (48GB total) was used to train the following configs:
- SingleModel02*
- Distillation03
