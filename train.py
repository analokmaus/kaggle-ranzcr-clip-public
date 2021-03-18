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
from configs_segmentation import *
from datasets import INPUT_DIR, load_dataframe
from metrics import MacroAverageAUC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--n_cpu", type=int, default=40,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only one fold")
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", type=int, default=1,
                        help="test time augmentation ")
    parser.add_argument("--gpu", nargs="+", default=[])
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)
    if len(opt.gpu) == 0:
        opt.gpu = None

    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    LOGGER = TorchLogger(
        export_dir / f'{cfg.name}_{get_time("%y%m%d%H%M")}.log', 
        log_items=log_items, file=~opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)
    config_info(cfg, logger=LOGGER)
    
    ''' Prepare data '''
    SEG_FLAG = cfg.dataset.__name__ == 'CLiPDatasetSegmentation'    
    train = pd.read_csv(INPUT_DIR/'train.csv')
    test = pd.read_csv(INPUT_DIR/'sample_submission.csv')
    splitter = cfg.splitter
    scores = []
    predictions = []
    outoffolds = np.zeros((len(train), cfg.num_classes), dtype=float)
    cv2.setNumThreads(0)
    seed_everything(cfg.seed, cfg.deterministic)

    test_data = cfg.dataset(
        df=test, transforms=cfg.transforms['test'], is_test=True, **cfg.dataset_params)
    test_loader = D.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=opt.n_cpu, pin_memory=True)
    eval_metric = MacroAverageAUC(reduce=False)

    if hasattr(splitter, 'split'):
        fold_iter = splitter.split(
            X=train,
            y=train[cfg.target_cols],
            groups=train['PatientID'])
    elif isinstance(splitter, (str, Path)):
        folds_csv = pd.read_csv(splitter)
        fold_iter = [
            (np.where(folds_csv['fold'] != fold)[0],
            np.where(folds_csv['fold'] == fold)[0])
            for fold in range(cfg.cv)
        ]

    '''
    Training
    '''
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        
        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== FOLD{fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        #####
        if 'use_annotations' in cfg.dataset_params.keys() and \
            cfg.dataset_params['use_annotations']:
            train_annotations = pd.read_csv(cfg.dataset_params['df_annotations'])
            train_fold = train_fold.loc[train_fold['StudyInstanceUID'].isin(
                train_annotations['StudyInstanceUID'].unique())]
            valid_fold = valid_fold.loc[valid_fold['StudyInstanceUID'].isin(
                train_annotations['StudyInstanceUID'].unique())]
        #####

        LOGGER(f'train positive: {train_fold[cfg.target_cols].values.mean(0)} ({len(train_fold)})')
        LOGGER(f'valid positive: {valid_fold[cfg.target_cols].values.mean(0)} ({len(valid_fold)})')

        train_data = cfg.dataset(
            df=train_fold, transforms=cfg.transforms['train'], is_test=False, **cfg.dataset_params)
        valid_data = cfg.dataset(
            df=valid_fold, transforms=cfg.transforms['test'], is_test=False, **cfg.dataset_params)

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=opt.n_cpu, pin_memory=True)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.n_cpu, pin_memory=True)

        model = cfg.model(**cfg.model_params)

        #####
        if cfg.pretrain_name is not None:
            weights = list(Path(f'results/{cfg.pretrain_name}').glob('fold?.pt'))
            LOGGER(f'Weights: \n{weights}')
            if len(weights) == cfg.cv: # multiple checkpoints in directory
                checkpoint = torch.load(
                    f'results/{cfg.pretrain_name}/fold{fold}.pt', map_location='cpu')
                LOGGER(f'Model: results/{cfg.pretrain_name}/fold{fold}.pt loaded.')
            else: # single checkpoint
                checkpoint = torch.load(weights[0], map_location='cpu')
                LOGGER(f'Model: {weights[0]} loaded.')
            if SEG_FLAG:
                fit_state_dict(checkpoint['model'], model.encoder)
                model.encoder.load_state_dict(checkpoint['model'], strict=False)
            elif cfg.model.__name__ == 'SegmentationAndClassification' and \
                cfg.pretrain_only_segmentation:
                    fit_state_dict(checkpoint['model'], model.segmentation_model)
                    model.segmentation_model.load_state_dict(
                        checkpoint['model'], strict=False)
            else:
                fit_state_dict(checkpoint['model'], model)
                model.load_state_dict(checkpoint['model'], strict=False)
            
        #####

        optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
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
            'resume': cfg.resume,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar
        }
        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        # trainer.ddp_average_loss = False
        # trainer.ddp_sync_batch_norm = False
        # trainer.debug = True
        trainer.fit(**FIT_PARAMS)

    '''
    Inference
    '''
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue # skip fold

        if SEG_FLAG:
            continue

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== FOLD{fold} =====')

        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(
            df=valid_fold, transforms=cfg.transforms['test'], is_test=False, 
            **cfg.dataset_params)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.n_cpu, pin_memory=True)

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(Path(f'results/{cfg.name}/fold{fold}.pt'), 'cpu')['model']
        fit_state_dict(checkpoint, model)
        model.load_state_dict(checkpoint)

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)

        if opt.tta > 1:
            prediction_fold = []
            for tta_i in range(opt.tta):
                prediction_fold.append(trainer.predict(
                    test_loader, progress_bar=opt.progress_bar, parallel='dp'))
            prediction_fold = np.stack(prediction_fold).mean(0)
        else:
            prediction_fold = trainer.predict(
                test_loader, progress_bar=opt.progress_bar, parallel='dp')
        outoffold = trainer.predict(
            valid_loader, progress_bar=opt.progress_bar, parallel='dp')

        predictions.append(prediction_fold)
        outoffolds[valid_idx] = outoffold

        metric_fold = eval_metric(
            valid_fold[cfg.target_cols].values, outoffold)
        metric_fold_mean = np.mean(metric_fold)
        scores.append([metric_fold_mean] + metric_fold)
        LOGGER('===== FOLDEND =====')
        del model, trainer; gc.collect()
        torch.cuda.empty_cache()

    '''
    Evaluation
    '''
    if SEG_FLAG:
        LOGGER('===== ALLEND =====')

    else:
        predictions = np.stack(predictions)
        np.save(export_dir/'prediction', predictions)
        np.save(export_dir/'outoffold', outoffolds)
        LOGGER(f'===== RESULTS =====')
        scores = pd.DataFrame(scores)
        scores.columns = ['MacroAverage'] + cfg.target_cols
        overall_mean = scores.MacroAverage.mean()
        overall_std = scores.MacroAverage.std()
        LOGGER(f'\n{scores.to_string()}')
        LOGGER(f'Overall: {overall_mean:.6f} + {overall_std:.6f}')
        LOGGER('===== ALLEND =====')
