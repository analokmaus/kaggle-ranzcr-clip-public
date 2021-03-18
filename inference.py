import argparse
from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy
from pprint import pprint
import random
import os
import gc
import time

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.utils.data as D

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict
from kuma_utils.utils import sigmoid

from configs2 import *
from datasets import *
from metrics import MacroAverageAUC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='SegAndCls12',
                        help="config name in configs.py")
    parser.add_argument("--n_cpu", type=int, default=40,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only one fold")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        # 'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    LOGGER = TorchLogger(
        export_dir / f'{cfg.name}_{get_time("%y%m%d%H%M")}.log', 
        log_items=log_items, file=False
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)
    config_info(cfg, logger=LOGGER)
    
    ''' Prepare data '''
    nih_filter = pd.read_csv('input/nih_files.csv')['StudyInstanceUID'].apply(
        lambda x: f'{Path(x).parents[1].name}/images/{Path(x).name}').values
    full_image = []
    full_image += [str(NIH_DIR/p) for p in nih_filter] # NIH 0:67468
    full_image += [str(p) for p in MIMIC_DIR.glob('**/*.jpg')] # MIMIC 67468:127437
    full_image += [str(p) for p in PADCHEST_DIR.glob('**/*.png')] # PadChest 127437:135049
    full_image = np.array(full_image)
    LOGGER(f'{len(full_image)} images found.')
    dummy_label = np.zeros((len(full_image), len(cfg.target_cols)))

    cv2.setNumThreads(0)
    seed_everything(cfg.seed, cfg.deterministic)
    pseudo_labels = []

    for fold in range(cfg.cv):

        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            # skip fold
            continue

        train_data = GeneralImageDataset(
            images=full_image,
            labels=dummy_label,
            transforms=cfg.transforms['test'],
        )
        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.n_cpu, pin_memory=True
        )

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(
            Path(f'results/{cfg.name}/fold{fold}.pt'), 'cpu')['model']
        fit_state_dict(checkpoint, model)
        model.load_state_dict(checkpoint)

        trainer = TorchTrainer(model, serial=f'fold{fold}')
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)
        pseudo_labels.append(sigmoid(trainer.predict(
            train_loader, progress_bar=opt.progress_bar, parallel='dp')))

        del model, trainer; gc.collect()
        torch.cuda.empty_cache()

        LOGGER('===== FOLDEND =====')
    
    pseudo_labels = np.stack(pseudo_labels)
    np.save(export_dir/'external_images', full_image)
    np.save(export_dir/'external_labels', pseudo_labels)