from pathlib import Path
from pprint import pformat
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch import ToTensor, ToTensorV2

from kuma_utils.torch import TorchLogger
from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import SimpleHook

from timm_latest import create_model
import segmentation_models_pytorch as smp
from models.custom_models import *

from datasets import *
from metrics import *
from utils import MultiStratifiedGroupKFold
from transforms import get_transforms
from models.segmentation_models import *
from losses import *

from configs2 import *


'''
Segmentation model
'''
class Segmentation10(Baseline):
    '''
    3 class EffNet-b7 512
    '''

    name = 'segmentation_10'


    segmentation_map = {
        'ETT - Abnormal': 0,
        'ETT - Borderline': 0,
        'ETT - Normal': 0,
        'NGT - Abnormal': 1,
        'NGT - Borderline': 1,
        'NGT - Incompletely Imaged': 1,
        'NGT - Normal': 1,
        'CVC - Abnormal': 2,
        'CVC - Borderline': 2,
        'CVC - Normal': 2,
        'Swan Ganz Catheter Present': 2,
    }

    dataset = CLiPDatasetSegmentation
    dataset_params = dict(
        df_annotations=INPUT_DIR/'train_annotations.csv',
        use_annotations=True,
        use_lungcontour=False,
        cmap=segmentation_map,
        annotation_style='line',
        annotation_size=60,
        grayscale=False
    )

    model = smp.Unet
    model_params = dict(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation='sigmoid'
    )

    num_epochs = 15
    criterion = smp.utils.losses.DiceLoss()
    eval_metric = IoU(threshold=0.5)
    hook = SimpleHook(evaluate_in_batch=True)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)

    img_size = 512
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }


class Segmentation13(Segmentation10):
    '''
    3 class ResNet200d 768
    '''

    name = 'segmentation_13'

    segmentation_map = {
        'ETT - Abnormal': 0,
        'ETT - Borderline': 0,
        'ETT - Normal': 0,
        'NGT - Abnormal': 1,
        'NGT - Borderline': 1,
        'NGT - Incompletely Imaged': 1,
        'NGT - Normal': 1,
        'CVC - Abnormal': 2,
        'CVC - Borderline': 2,
        'CVC - Normal': 2,
        'Swan Ganz Catheter Present': 2
    }

    dataset_params = dict(
        df_annotations=INPUT_DIR/'train_annotations.csv',
        use_annotations=True,
        use_lungcontour=False,
        cmap=segmentation_map,
        annotation_style='line',
        annotation_size=60,
        grayscale=False
    )
    model_params = dict(
        encoder_name="resnet200d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation='sigmoid'
    )

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }


class Segmentation15(Segmentation10):
    '''
    768 3 class EffNet-b7
    '''

    name = 'segmentation_15'

    batch_size = 12
    optimizer_params = dict(lr=2e-4)

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }
