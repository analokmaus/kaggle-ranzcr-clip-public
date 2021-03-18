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
from models.segmentation_models import ResNet200dEncoder
from losses import *


USE_PRETRAINED = True
smp.encoders.encoders["resnet200d"] = {
    "encoder": ResNet200dEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}


def config_info(cfg, logger=None):

    def _print(text):
        if logger is None:
            print(text)
        else:
            logger(text)

    items = [
        'name', 'pretrain_name',
        'cv', 'num_epochs', 'batch_size', 'seed',
        'dataset', 'dataset_params', 'img_size', 'num_classes', 'transforms', 'splitter',
        'model', 'model_params', 'optimizer', 'optimizer_params',
        'scheduler', 'scheduler_params', 'batch_scheduler', 'scheduler_target',
        'criterion', 'eval_metric', 'monitor_metrics',
        'amp', 'parallel', 'hook', 'callbacks', 'deterministic',
    ]
    _print('===== CONFIG =====')
    for key in items:
        try:
            val = getattr(cfg, key)
            if isinstance(val, (type, types.FunctionType)):
                val = val.__name__ + '(*)'
            if isinstance(val, (dict, list)):
                val = '\n'+pformat(val, compact=True, indent=2)
            _print(f'{key}: {val}')
        except:
            _print(f'{key}: ERROR')
    _print(f'===== CONFIGEND =====')


class Baseline:

    # Settings
    name = 'baseline'
    pretrain_name = None
    pretrain_only_encoder = False
    pretrain_only_segmentation = False
    cv = 5
    num_epochs = 21
    batch_size = 48
    seed = 2021
    resume = False

    # Dataset
    dataset = CLiPDataset
    dataset_params = dict()
    img_size = 512
    transforms = {
        'train': get_transforms(img_size, 'basic'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }
    splitter = MultiStratifiedGroupKFold(n_splits=cv, random_state=seed)
    target_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    num_classes = len(target_cols)

    # Training
    model = create_model
    model_params = dict(
        model_name='resnet200d',
        pretrained=USE_PRETRAINED,
        num_classes=11
    )
    lr = 5e-4
    optimizer = optim.Adam
    optimizer_params = dict(lr=lr)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = MacroAverageAUC().torch
    monitor_metrics = []
    amp = True
    parallel = 'ddp'
    deterministic = False
    max_grad_norm = 1000000
    hook = SimpleHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=True),
        SaveSnapshot()
    ]


'''
Segmentation + Classification Model
'''

class SegAndCls12(Baseline):
    '''
    768 RGB 3 class ResNet200d
    '''

    name = 'seg_and_cls_12'
    pretrain_name = 'segmentation_13'
    pretrain_only_segmentation = True

    batch_size = 24
    num_epochs = 15

    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=15, eta_min=1e-6)

    model = SegmentationAndClassification
    model_params = dict(
        segmentation_model='resnet200d',
        segmentation_classes=3,
        classification_model='resnet18',
        classification_classes=11,
        pretrained=USE_PRETRAINED,
        return_mask=False,
        freeze_segmentation=False
    )
    criterion = FocalLoss()
    
    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }


class SegAndCls12external6(SegAndCls12):

    name = 'seg_and_cls_12_external_6'
    pretrain_name = 'pretrain_student_08_l'
    pretrain_only_segmentation = False

    num_epochs = 10
    optimizer_params = dict(lr=1e-4)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=10, eta_min=1e-6)

    criterion = FocalLoss2(
        smoothing=[
            0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01,
            0.0, 0.0, 0.01, 0.0]
    )

    callbacks = [
        EarlyStopping(patience=10, maximize=True),
        SaveSnapshot()
    ]

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong_plus', remove_border=True),
        'test': get_transforms(img_size, 'basic', remove_border=True),
        'tta': get_transforms(img_size, 'basic', remove_border=True),
    }


class SegAndCls14(Baseline):

    name = 'seg_and_cls_14'
    pretrain_name = 'segmentation_15'
    pretrain_only_segmentation = True

    batch_size = 12
    num_epochs = 15

    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=15, eta_min=1e-6)

    model = SegmentationAndClassification
    model_params = dict(
        segmentation_model='efficientnet-b7',
        segmentation_classes=3,
        classification_model='tf_efficientnet_b0',
        classification_classes=11,
        pretrained=USE_PRETRAINED,
        return_mask=False,
        freeze_segmentation=False,
        concat_original=False,
    )
    criterion = FocalLoss()

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }


class SegAndCls14external2(SegAndCls14):

    name = 'seg_and_cls_14_external_2'
    pretrain_name = 'pretrain_student_09'
    pretrain_only_segmentation = False
    
    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong_plus', remove_border=True),
        'test': get_transforms(img_size, 'basic', remove_border=True),
        'tta': get_transforms(img_size, 'basic', remove_border=True),
    }

    num_epochs = 5
    optimizer_params = dict(lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    monitor_metrics = [MacroAveragePRAUC().torch]
    callbacks = [
        SaveEveryEpoch(),
        SaveAllSnapshots()
    ]


'''
Vanilla CNN
'''
class SingleModel02(Baseline):
    '''
    Don't use this config to train NFNet-F1.
    Please copy pretrained weights from ().
    '''

    name = 'nakama_nfnetf1'
    pretrain_name = None

    batch_size = 16
    num_epochs = 10
    model = CustomNFNet
    model_params = dict(
        model_name='dm_nfnet_f1',
        pretrained=USE_PRETRAINED,
        num_classes=11
    )
    max_grad_norm = 1000

    criterion = FocalLoss2(
        smoothing=[
            0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01,
            0.0, 0.0, 0.01, 0.0]
    )
    callbacks = [
        EarlyStopping(patience=5, maximize=True),
        SaveSnapshot()
    ]

    optimizer = optim.Adam
    optimizer_params = dict(lr=1e-4)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=10, eta_min=1e-6)

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong_plus', 
                                norm_mean=[0.485, 0.456, 0.406],
                                norm_std=[0.229, 0.224, 0.225],
                                remove_border=True),
        'test': get_transforms(img_size, 'basic', 
                               norm_mean=[0.485, 0.456, 0.406],
                               norm_std=[0.229, 0.224, 0.225],
                               remove_border=True),
        'tta': get_transforms(img_size, 'basic', 
                              norm_mean=[0.485, 0.456, 0.406],
                              norm_std=[0.229, 0.224, 0.225], remove_border=True),
    }


class SingleModel02external0(SingleModel02):

    name = 'single_model_02_external_0'
    pretrain_name = 'distillation_03'
    
    num_epochs = 6
    optimizer_params = dict(lr=2e-5)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=6, eta_min=1e-6)

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong_plus', remove_border=True),
        'test': get_transforms(img_size, 'basic', remove_border=True),
        'tta': get_transforms(img_size, 'basic', remove_border=True),
    }

    monitor_metrics = [MacroAveragePRAUC().torch]
    callbacks = [
        SaveEveryEpoch(),
        SaveAllSnapshots()
    ]
