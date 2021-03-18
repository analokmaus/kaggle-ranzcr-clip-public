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

from configs_external import *
from datasets import INPUT_DIR, NIH_DIR, MIMIC_DIR, PADCHEST_DIR
from metrics import MacroAverageAUC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='PretrainChest00',
                        help="config name in configs.py")
    parser.add_argument("--n_cpu", type=int, default=40,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only one fold")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
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
        log_items=log_items, file=True
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)
    config_info(cfg, logger=LOGGER)
    
    ''' Prepare data '''
    _full_image = np.load(Path('results')/f'{cfg.external_config}/external_images.npy')
    _full_label = np.load(Path('results')/f'{cfg.external_config}/external_labels.npy')
    full_image = []
    full_label = []
    for d in cfg.external_dataset:
        if d == 'MIMIC':
            full_image.append(_full_image[67468:127437])
            full_label.append(_full_label[:, 67468:127437])
        elif d == 'NIH':
            full_image.append(_full_image[0:67468])
            full_label.append(_full_label[:, 0:67468])
        elif d == 'PadChest':
            full_image.append(_full_image[127437:135049])
            full_label.append(_full_label[:, 127437:135049])
        else:
            raise ValueError(f'Invalid dataset name {d}.')
    full_image = np.concatenate(full_image)
    full_label = np.concatenate(full_label, axis=1)
    LOGGER(f'Pseudo labels from {cfg.external_config}({cfg.external_dataset}) loaded.')
    

    cv2.setNumThreads(0)
    seed_everything(cfg.seed, cfg.deterministic)

    '''
    Training
    '''
    for fold in range(cfg.cv):

        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            # skip fold
            continue
        
        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        label_fold = full_label[fold].copy()
        image_fold = full_image.copy()

        if cfg.sample_certainty != 0:
            label_fold_hard = (label_fold >= 0.5).astype(float)
            uncertainty = np.abs(label_fold_hard - label_fold).sum(1)
            sample_k = int(label_fold.shape[0] * np.abs(cfg.sample_certainty))
            if cfg.sample_certainty > 0: # sample certain samples
                certainty_mask = np.argsort(uncertainty)[:sample_k]
            else: # sample uncertain samples
                certainty_mask = np.argsort(uncertainty)[-sample_k:]
            label_fold = label_fold[certainty_mask, :]
            image_fold = image_fold[certainty_mask]

        train_data = cfg.dataset(
            images=image_fold,
            labels=label_fold,
            transforms=cfg.transforms['train'],
            **cfg.dataset_params
        )
        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=opt.n_cpu, pin_memory=True
        )

        LOGGER(f'===== EXTERNAL FOLD {fold} =====')
        LOGGER(f'dataset size: {len(image_fold)}')

        model = cfg.model(**cfg.model_params)

        #####
        if cfg.pretrain_segmentation is not None:
            weights = list(Path(f'results/{cfg.pretrain_segmentation}').glob('*.pt'))
            if len(weights) == cfg.cv:  # multiple checkpoints in directory
                checkpoint = torch.load(
                    f'results/{cfg.pretrain_segmentation}/fold{fold}.pt', map_location='cpu')
            else:  # single checkpoint
                checkpoint = torch.load(weights[0], map_location='cpu')
            fit_state_dict(checkpoint['model'], model.segmentation_model)
            model.segmentation_model.load_state_dict(checkpoint['model'], strict=False)
            LOGGER(f'Student: {cfg.pretrain_segmentation} loaded.')
        #####

        optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': None,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': cfg.callbacks,
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'max_grad_norm': cfg.max_grad_norm,
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar
        }
        trainer = TorchTrainer(model, serial=f'fold{fold}')
        trainer.fit(**FIT_PARAMS)

        del model, trainer; gc.collect()
        torch.cuda.empty_cache()

        LOGGER('===== FOLDEND =====')
