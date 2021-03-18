import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import ast
import torch
import torch.utils.data as D


INPUT_DIR = Path('../data/ranzcr-clip/')
NIH_DIR = INPUT_DIR/'nih_chestxray'
MIMIC_DIR = INPUT_DIR/'mimic/mimic'
PADCHEST_DIR = INPUT_DIR/'padchest/padchest/padchest'


def load_dataframe(arg):
    if isinstance(arg, (str, Path)):
        return pd.read_csv(arg)
    elif isinstance(arg, pd.DataFrame):
        return arg
    else:
        return None


class CLiPDataset(D.Dataset):

    def __init__(self, df, transforms=None, is_test=False, mixup=False, mixup_alpha=0.4,
                 target_cols=None, grayscale=False, input_dir=INPUT_DIR):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.mixup = mixup
        self.alpha = mixup_alpha
        self.grayscale = grayscale
        if self.is_test:
            self.image_dir = input_dir / 'test'
            self.mixup = False
        else:
            self.image_dir = input_dir / 'train'
        if target_cols is None:
            self.target_cols = [
                'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                'Swan Ganz Catheter Present'
            ]
        else:
            self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def _load_image_label(self, idx):
        studyid = self.df.iloc[idx]['StudyInstanceUID']
        img_path = str(self.image_dir / f'{studyid}.jpg')
        if self.grayscale:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[:, :, None]
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        label = torch.tensor(self.df.iloc[idx][self.target_cols]).float()
        return image, label

    def __getitem__(self, idx):
        image1, label1 = self._load_image_label(idx)
        if self.mixup:
            idx2 = np.random.randint(0, len(self))
            lam = np.random.beta(self.alpha, self.alpha)
            image2, label2 = self._load_image_label(idx2)
            image1 = lam * image1 + (1 - lam) * image2
            label1 = lam * label1 + (1 - lam) * label2
        return image1, label1


class CLiPDatasetSegmentation(D.Dataset):

    def __init__(self, df, df_annotations, df_lungcontour=None,
                 use_annotations=False, use_lungcontour=False,
                 annotation_size=50, annotation_style='point', cmap=None,
                 grayscale=False, return_label=False, 
                 transforms=None, is_test=False, input_dir=INPUT_DIR):
        super().__init__()
        self.df = df
        self.df_annotations = load_dataframe(df_annotations)
        self.df_lungcontour = load_dataframe(df_lungcontour)
        self.use_annot = use_annotations
        self.use_lung = use_lungcontour
        self.annot_size = annotation_size
        self.annot_style = annotation_style
        self.grayscale = grayscale
        self.file_names = df['StudyInstanceUID'].values
        self.target_cols = [
            'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
            'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
            'Swan Ganz Catheter Present'
        ]
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = {
                'ETT - Abnormal': 0,
                'ETT - Borderline': 1,
                'ETT - Normal': 2,
                'NGT - Abnormal': 3,
                'NGT - Borderline': 4,
                'NGT - Incompletely Imaged': 5,
                'NGT - Normal': 6,
                'CVC - Abnormal': 7,
                'CVC - Borderline': 8,
                'CVC - Normal': 9,
                'Swan Ganz Catheter Present': 10,
            }
        if self.use_lung and 'lung' not in self.cmap.keys():
            self.cmap['lung'] = max(self.cmap.values()) + 1
        self.num_classes = max(self.cmap.values()) + 1
        self.labels = df[self.target_cols].values
        self.return_label = return_label
        self.transforms = transforms
        if is_test:
            self.image_dir = input_dir / 'test'
        else:
            self.image_dir = input_dir / 'train'

    def __len__(self):
        return len(self.df)

    def _load_image_draw_mask(self, idx):
        file_name = self.file_names[idx]
        query_string = f"StudyInstanceUID == '{file_name}'"
        file_path = str(self.image_dir / f'{file_name}.jpg')
        if self.grayscale:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[:, :, None]
        else:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(
            (image.shape[0], image.shape[1], self.num_classes), dtype=np.uint8)

        if self.use_annot:
            annotations = self.df_annotations.query(query_string)
            for i, row in annotations.iterrows():
                label = row["label"]
                data = np.array(ast.literal_eval(row["data"]))
                _layer = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8)
                if self.annot_style == 'point' or label == 'CVC Tips':
                    for d in data:
                        cv2.circle(
                            img=_layer,
                            center=(d[0], d[1]),
                            radius=self.annot_size,
                            thickness=-1,
                            color=1)
                elif self.annot_style == 'line':
                    cv2.polylines(
                        _layer, [data], False,
                        thickness=self.annot_size,
                        color=1)
                mask[:, :, self.cmap[label]] = mask[:,
                                                    :, self.cmap[label]] + _layer

        if self.use_lung:
            annotations = self.df_lungcontour.query(query_string)
            ctr_left = np.array(ast.literal_eval(
                annotations['left_lung_contour'].values[0]))
            ctr_right = np.array(ast.literal_eval(
                annotations['right_lung_contour'].values[0]))
            _layer = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(
                _layer, pts=[ctr_left], color=1)
            cv2.fillPoly(
                _layer, pts=[ctr_right], color=1)
            mask[:, :, self.cmap['lung']] = mask[:,
                                                 :, self.cmap['lung']] + _layer

        mask = np.clip(mask, 0, 1)

        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load_image_draw_mask(idx)
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        if self.return_label:
            label = torch.tensor(self.labels[idx]).float()
            return image, mask.permute(2, 0, 1), label
        else:
            return image, mask.permute(2, 0, 1)


'''
External
'''
class GeneralImageDataset(D.Dataset):

    def __init__(self, images, labels=None, transforms=None, grayscale=False, hard_label=False):
        super().__init__()
        self.images = images
        self.labels = labels
        self.grayscale = grayscale
        self.transforms = transforms
        self.hard_label = hard_label

    def __len__(self):
        return len(self.images)

    def _load_image_label(self, idx):
        img_path = str(self.images[idx])
        if self.grayscale:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[:, :, None]
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        if self.labels is None:
            label = torch.tensor(-1).float()
        else:
            label = torch.tensor(self.labels[idx]).float()
            if self.hard_label:
                label = (label > 0.5).float()
        return image, label

    def __getitem__(self, idx):
        image1, label1 = self._load_image_label(idx)
        return image1, label1
