import numpy as np
import random
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensor, ToTensorV2


class RemoveBlackBorder(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):
        super(RemoveBlackBorder, self).__init__(always_apply, p)

    def apply(self, img, **params):
        mask = img[:, :, 0] > 0
        return img[np.ix_(mask.any(1), mask.any(0))]

    def get_transform_init_args_names(self):
        return ()


def get_transforms(
    img_size, profile='basic', 
    norm_mean=0.4824, norm_std=0.225, remove_border=False, clahe=False, additional_targets=None):

    if profile == 'basic':
        ts =[
            A.Resize(img_size, img_size),
        ]
    elif profile == 'strong':
        ts = [
            A.RandomResizedCrop(
                img_size, img_size, scale=(0.9, 1), p=1),
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
            A.Cutout(max_h_size=int(img_size * 0.1),
                     max_w_size=int(img_size * 0.1), num_holes=5, p=0.5),
        ]
    elif profile == 'strong_plus':
        ts = [
            A.RandomResizedCrop(
                img_size, img_size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.RandomGamma(p=0.5), 
            A.OneOf([
                A.OpticalDistortion(),
                A.IAAPiecewiseAffine(),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(),
                A.MotionBlur(blur_limit=(3, 5)),
            ], p=0.1),
            A.Cutout(max_h_size=int(img_size * 0.1),
                     max_w_size=int(img_size * 0.1), num_holes=5, p=0.5),
        ]
    elif profile == 'strong_noflip':
        ts = [
            A.RandomResizedCrop(
                img_size, img_size, scale=(0.9, 1), p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.RandomGamma(p=0.5), 
            A.OneOf([
                A.OpticalDistortion(),
                A.IAAPiecewiseAffine(),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(),
                A.MotionBlur(blur_limit=(3, 5)),
            ], p=0.1),
            A.Cutout(max_h_size=int(img_size * 0.1),
                     max_w_size=int(img_size * 0.1), num_holes=5, p=0.5),
        ]
    
    if remove_border:
        ts = [RemoveBlackBorder()] + ts
    
    if clahe:
        ts = ts + [A.CLAHE(always_apply=True, p=1.0)]
    
    ts = ts + [A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()]

    if additional_targets is None:
        return A.Compose(ts)
    else:
        return A.Compose(ts, additional_targets=additional_targets)
