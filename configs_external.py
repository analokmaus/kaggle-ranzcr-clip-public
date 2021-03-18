from pathlib import Path
from pprint import pformat
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import GroupKFold, KFold

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

from configs2 import *

'''
Pretrain on external dataset in a student-teacher architecture
'''
class PretrainStudent04(Baseline):

    name = 'pretrain_student_04'
    pretrain_name = 'seg_and_cls_12'
    pretrain_segmentation = 'segmentation_13'

    model = SegmentationAndClassification
    model_params = dict(
        segmentation_model='resnet200d',
        segmentation_classes=3,
        classification_model='resnet18',
        classification_classes=11,
        pretrained=USE_PRETRAINED,
        return_mask=False,
        return_feature=False,
    )

    dataset = GeneralImageDataset
    dataset_params = dict(
        hard_label=False
    )
    sample_certainty = 0.0
    external_config = 'seg_and_cls_12'
    external_dataset = ['NIH', 'MIMIC', 'PadChest']

    batch_size = 24
    num_epochs = 5
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = nn.BCEWithLogitsLoss()
    hook = SimpleHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=False, target='train_metric'),
        SaveSnapshot()
    ]

    lr = 2e-4
    optimizer = optim.Adam
    optimizer_params = dict(lr=lr)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=5, eta_min=1e-6)

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong'),
        'test': get_transforms(img_size, 'basic'),
        'tta': get_transforms(img_size, 'basic'),
    }


class PretrainStudent08l(PretrainStudent04):

    name = 'pretrain_student_08_l'
    external_config = 'seg_and_cls_12'
    external_dataset = ['MIMIC', 'NIH']
    pretrain_name = None
    pretrain_segmentation = 'segmentation_13'
    num_epochs = 5
    criterion = nn.MSELoss()
    dataset_params = dict(
        hard_label=False
    )


class PretrainStudent09(PretrainStudent04):

    name = 'pretrain_student_09'
    external_config = 'seg_and_cls_14'
    external_dataset = ['NIH']
    pretrain_name = None
    pretrain_segmentation = 'segmentation_15'
    num_epochs = 4
    dataset_params = dict(
        hard_label=False
    )
    external_dataset = ['MIMIC']
    criterion = nn.MSELoss()

    batch_size = 12
    num_epochs = 4

    model_params = dict(
        segmentation_model='efficientnet-b7',
        segmentation_classes=3,
        classification_model='tf_efficientnet_b0',
        classification_classes=11,
        pretrained=USE_PRETRAINED
    )


class Distillation03(Baseline):

    name = 'distillation_03'

    pretrain_name = None
    pretrain_segmentation = None

    batch_size = 16
    num_epochs = 5
    cv = 5

    dataset = GeneralImageDataset
    dataset_params = dict(
        hard_label=False
    )
    external_config = 'nakama_nfnetf1'
    external_dataset = ['MIMIC']

    model = CustomNFNet
    model_params = dict(
        model_name='dm_nfnet_f1',
        pretrained=USE_PRETRAINED,
        num_classes=11
    )
    max_grad_norm = 1000

    criterion = nn.MSELoss()
    eval_metric = nn.MSELoss()
    callbacks = [
        EarlyStopping(patience=5, maximize=False, target='train_metric'),
        SaveSnapshot()
    ]

    optimizer = optim.Adam
    optimizer_params = dict(lr=1e-4)
    scheduler = CosineAnnealingLR
    scheduler_params = dict(T_max=4, eta_min=1e-6)

    img_size = 768
    transforms = {
        'train': get_transforms(img_size, 'strong', remove_border=True),
        'test': get_transforms(img_size, 'basic', remove_border=True),
        'tta': get_transforms(img_size, 'basic', remove_border=True),
    }
