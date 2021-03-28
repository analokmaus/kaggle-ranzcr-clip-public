# ====================================================
# Directory settings
# ====================================================
import warnings
from apex import amp
import timm
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
import cv2
from functools import partial
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import scipy as sp
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
import shutil
import random
import time
import math
import ast
import sys
import gc
import os

OUTPUT_DIR = '../output/nfnet_f1_stage23/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = '../input/ranzcr-clip-catheter-line-classification/train'
TEACHER_DIR = '../output/nfnet_f1_stage1/'

# ====================================================
# CFG
# ====================================================


class CFG:
    debug = False
    apex = True
    print_freq = 100
    num_workers = 8
    model_name = 'dm_nfnet_f1'
    weights = [0.5, 1]
    size = 768
    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    scheduler = 'CosineAnnealingWarmRestarts'
    pre_epochs = 5
    post_epochs = 10  # 15
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    #pre_T_max=5 # CosineAnnealingLR
    #post_T_max=15 # CosineAnnealingLR
    pre_T_0 = 5  # CosineAnnealingWarmRestarts
    post_T_0 = 5  # CosineAnnealingWarmRestarts
    lr = 1e-4  # 2e-4
    min_lr = 1e-6
    pre_batch_size = 8  # 12 # 24
    post_batch_size = 8  # 12 # 24
    weight_decay = 1e-6
    gradient_accumulation_steps = 1  # 2
    max_grad_norm = 1000
    alpha = 1
    gamma = 2
    smoothing = [0.0, 0.0, 0.01,
                 0.0, 0.0, 0.0, 0.01,
                 0.0, 0.0, 0.01,
                 0.0]
    seed = 416
    target_size = 11
    target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                   'Swan Ganz Catheter Present']
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]  # [0, 1, 2, 3, 4]
    train = True


if CFG.debug:
    CFG.epochs = 1

# ====================================================
# Library
# ====================================================


warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================
# Utils
# ====================================================


def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv(
    '../input/ranzcr-clip-catheter-line-classification/train.csv')
folds = pd.read_csv(f'../preprocess/folds_seed{CFG.seed}.csv')
train_annotations = pd.read_csv(
    '../input/ranzcr-clip-catheter-line-classification/train_annotations.csv')

if CFG.debug:
    CFG.epochs = 1
    folds = folds.sample(n=1000, random_state=CFG.seed)

# ====================================================
# Dataset
# ====================================================
COLOR_MAP = {'ETT - Abnormal': (255, 0, 0),
             'ETT - Borderline': (0, 255, 0),
             'ETT - Normal': (0, 0, 255),
             'NGT - Abnormal': (255, 255, 0),
             'NGT - Borderline': (255, 0, 255),
             'NGT - Incompletely Imaged': (0, 255, 255),
             'NGT - Normal': (128, 0, 0),
             'CVC - Abnormal': (0, 128, 0),
             'CVC - Borderline': (0, 0, 128),
             'CVC - Normal': (128, 128, 0),
             'Swan Ganz Catheter Present': (128, 0, 128),
             }


class TrainDataset(Dataset):
    def __init__(self, df, df_annotations, use_annot=False, annot_size=50, transform=None):
        self.df = df
        self.df_annotations = df_annotations
        self.use_annot = use_annot
        self.annot_size = annot_size
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[CFG.target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = torch.tensor(self.labels[idx]).float()
        if self.use_annot:
            image_annot = image.copy()
            query_string = f"StudyInstanceUID == '{file_name}'"
            df = self.df_annotations.query(query_string)
            for i, row in df.iterrows():
                label = row["label"]
                data = np.array(ast.literal_eval(row["data"]))
                for d in data:
                    image_annot[d[1]-self.annot_size//2:d[1]+self.annot_size//2,
                                d[0]-self.annot_size//2:d[0]+self.annot_size//2,
                                :] = COLOR_MAP[label]
            if self.transform:
                augmented = self.transform(
                    image=image, image_annot=image_annot)
                image = augmented['image']
                image_annot = augmented['image_annot']
            return image, image_annot, labels
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, labels

# ====================================================
# Transforms
# ====================================================


def get_transforms(*, data):

    if data == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.size, CFG.size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.OneOf([
                A.OpticalDistortion(),
                A.IAAPiecewiseAffine(),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(),
                A.MotionBlur(blur_limit=(3, 5)),
            ], p=0.1),
            A.Cutout(max_h_size=int(CFG.size * 0.1),
                     max_w_size=int(CFG.size * 0.1), num_holes=5, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], additional_targets={'image_annot': 'image'})

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

#from matplotlib import pyplot as plt

#train_dataset = TrainDataset(train, transform=get_transforms(data='train'))

#for i in range(5):
#    image, label = train_dataset[i]
#    plt.imshow(image[0])
#    plt.title(f'label: {label}')
#    plt.show()

# ====================================================
# MODEL
# ====================================================


class CustomNFNet(nn.Module):
    def __init__(self, model_name='dm_nfnet_f0', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.fc.in_features
        self.model.head.global_pool = nn.Identity()
        self.model.head.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output


# ====================================================
# Loss
# ====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=CFG.alpha, gamma=CFG.gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss


class SmoothFocalwLogits(nn.Module):
    def __init__(self, reduction='mean', smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothFocalwLogits._smooth(targets, self.smoothing)
        loss = FocalLoss()(inputs, targets)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def ranzcr_smooth_focal_w_logits(inputs, targets, reduction='mean', smoothing=CFG.smoothing):
    losses = torch.zeros(11).cuda()
    for i in range(11):
        loss_i = SmoothFocalwLogits(reduction, smoothing[i])(
            inputs[:, i], targets[:, i])
        losses[i] = loss_i
    loss = torch.mean(losses)
    return loss


class CustomLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, teacher_features, features, y_pred, labels):
        consistency_loss = nn.MSELoss()(teacher_features.view(-1), features.view(-1))
        #cls_loss = nn.BCEWithLogitsLoss()(y_pred, labels)
        cls_loss = ranzcr_smooth_focal_w_logits(y_pred, labels)
        loss = self.weights[0] * consistency_loss + self.weights[1] * cls_loss
        return loss

# ====================================================
# Helper functions
# ====================================================


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def pre_train_fn(train_loader, teacher_model, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, images_annot, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            teacher_features, _, _ = teacher_model(images_annot.to(device))
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        features, _, y_preds = model(images)
        loss = criterion(teacher_features, features, y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(
                      epoch+1, step, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,
                      remain=timeSince(start, float(step+1)/len(train_loader)),
                      grad_norm=grad_norm,
                      lr=scheduler.get_lr()[0],
                  ))
    return losses.avg


def post_train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        features, _, y_preds = model(images)
        loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(
                      epoch+1, step, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,
                      remain=timeSince(start, float(step+1)/len(train_loader)),
                      grad_norm=grad_norm,
                      lr=scheduler.get_lr()[0],
                  ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            _, _, y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                      step, len(valid_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,
                      remain=timeSince(start, float(step+1)/len(valid_loader)),
                  ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# Train loop
# ====================================================


def pre_train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} pre-training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_folds = train_folds[train_folds['StudyInstanceUID'].isin(
        train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)
    #valid_folds = valid_folds[valid_folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = TrainDataset(train_folds, train_annotations, use_annot=True,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, train_annotations, use_annot=False,
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.pre_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.pre_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=CFG.pre_T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.pre_T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    teacher_model = CustomNFNet(CFG.model_name, pretrained=False)
    teacher_path = TEACHER_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth'
    teacher_model.load_state_dict(torch.load(teacher_path)['model'])
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    teacher_model.to(device)

    model = CustomNFNet(CFG.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr,
                     weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    if CFG.apex:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    train_criterion = CustomLoss(weights=CFG.weights)
    valid_criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG.pre_epochs):

        start_time = time.time()

        # train
        avg_loss = pre_train_fn(train_loader, teacher_model, model,
                                train_criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(
            valid_loader, model, valid_criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')
        LOGGER.info(
            f'Epoch {epoch+1} - Score: {score:.6f}  Scores: {np.round(scores, decimals=6)}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.6f} Model')
            torch.save({'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), },
                       OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss_pre.pth')

    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    gc.collect()


def post_train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} post-training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = TrainDataset(train_folds, train_annotations, use_annot=False,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, train_annotations, use_annot=False,
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.post_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.post_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=CFG.post_T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.post_T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomNFNet(CFG.model_name, pretrained=False)

    states = torch.load(
        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss_pre.pth')
    model.load_state_dict(states['model'])
    model.to(device)
    optimizer = Adam(model.parameters(), lr=CFG.lr,
                     weight_decay=CFG.weight_decay, amsgrad=False)
    #optimizer.load_state_dict(states['optimizer'])
    #for state in optimizer.state.values():
    #    for k, v in state.items():
    #        if isinstance(v, torch.Tensor):
    #            state[k] = v.to('cuda')
    scheduler = get_scheduler(optimizer)
    #scheduler.load_state_dict(states['scheduler'])

    if CFG.apex:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    #criterion = nn.BCEWithLogitsLoss()
    criterion = ranzcr_smooth_focal_w_logits

    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG.post_epochs):

        start_time = time.time()

        # train
        avg_loss = post_train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')
        LOGGER.info(
            f'Epoch {epoch+1} - Score: {score:.6f}  Scores: {np.round(scores, decimals=6)}')

        if score > best_score:
            best_score = score
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.6f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.6f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')

    check_point = torch.load(
        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
    for c in [f'pred_{c}' for c in CFG.target_cols]:
        valid_folds[c] = np.nan
    valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']

    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds

# ====================================================
# main
# ====================================================


def main():
    """
    Prepare: 1.train  2.folds
    """

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
        labels = result_df[CFG.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(
            f'Score: {score:<.6f}  Scores: {np.round(scores, decimals=6)}')
        return score, scores

    if CFG.train:
        # train
        score_list = []
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                pre_train_loop(folds, fold)
                _oof_df = post_train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                score, _ = get_result(_oof_df)
                score_list.append(score)
        # CV result
        LOGGER.info(f"========== CV ==========")
        _, _ = get_result(oof_df)
        # CV (mean & std) result
        LOGGER.info(f"========== CV (mean & std) ==========")
        overall_mean = np.mean(score_list)
        overall_std = np.std(score_list)
        LOGGER.info(f'Overall: {overall_mean:.6f} + {overall_std:.6f}')
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)


if __name__ == '__main__':
    main()
