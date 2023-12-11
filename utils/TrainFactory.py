import sys
import os
import argparse
import logging
import logging.config
import yaml
import pathlib
import builtins
import socket
import time
import random
import numpy as np
import torch
import logging
import nrrd
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data
import wandb
import torch.nn.functional as F
from torch import nn
from os import path
from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.HaN import HaN
from libs.losses.LossFactory import LossFactory
from libs.losses.LossFactory import *
from libs.models.ModelFactory import ModelFactory
from libs.optimizers.OptimizerFactory import OptimizerFactory
from libs.schedulers.SchedulerFactory import SchedulerFactory
from utils.AugmentFactory import *
from utils.EvalFactory import Eval as Evaluator
from datasets.label_dict import LABEL_dict, Anchor_dict   # from datasets/label_dict.py
eps = 1e-10


class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}
        self.scaler = torch.cuda.amp.GradScaler()

        # num_classes = len(self.config.data_loader.labels)
        # if 'Jaccard' in self.config.loss.name or num_classes == 2:
        num_classes =  len(Anchor_dict) if self.config.experiment.name == 'Anchor' else len(LABEL_dict)
        self.num_classes = num_classes
        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1

        self.model = ModelFactory(model_name, num_classes, in_ch).get().cuda(self.config.device)
        for m in self.model.modules():
            for child in m.children():
                if type(child) == torch.nn.BatchNorm3d:
                    m.eval()

        
        wandb.watch(self.model, log_freq=10)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
            sched_name,
            self.optimizer,
            milestones=sched_milestones,
            gamma=sched_gamma,
            mode='max',
            verbose=True,
            patience=10
        ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, classes=num_classes)

        # load evaluator
        self.evaluator = Evaluator(classes=num_classes)
        self.train_dataset = HaN(
            config = self.config,
            splits='train',
            transform=tio.Compose([
                # tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                # self.config.data_loader.preprocessing,
                self.config.data_loader.augmentations,
            ]),
            sampler=self.config.data_loader.sampler_type
        )
        self.val_dataset = HaN(
            config = self.config,
            splits='val',
            # transform=self.config.data_loader.preprocessing,
        )
        self.test_dataset = HaN(
            config = self.config,
            splits='test',
            # transform=self.config.data_loader.preprocessing,
        )

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.experiment.name, 'train', self.config.title, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.config.title,
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)

    def load(self):
        path = self.config.trainer.checkpoint
        logging.info(f'Loading checkpoint from {path}')
        state = torch.load(path)

        if 'title' in state.keys():
            # check that the title headers (without the hash) is the same
            self_title_header = self.config.title[:-11]
            load_title_header = state['title'][:-11]
            if self_title_header == load_title_header:
                self.config.title = state['title']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['state_dict'])
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def extract_data_from_feature(self, feature):
        ct_volume = feature['ct'][tio.DATA].float().cuda(self.config.device)
        # mr_volume = feature['mr'][tio.DATA].float().cuda()
        gt = feature['label'][tio.DATA].float().cuda(self.config.device)
        # volume = volume/255. # normalization
        return ct_volume, gt

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader
        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt = self.extract_data_from_feature(d)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = self.model(images) # pred shape B, C(N), H, W, D
                preds_soft = F.softmax(preds, dim=1)
                # 이미 여기서 thread: [23,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size
                gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=self.num_classes)
                gt_onehot = gt_onehot.unsqueeze(0)
                gt_onehot = torch.movedim(gt_onehot, -1, 1)
                assert preds_soft.ndim == gt_onehot.ndim, f'Gt and output dimensions are not the same before loss. {preds_soft.ndim} vs {gt_onehot.ndim}'
                
                if self.loss.names[0] == 'Dice3DLoss':
                    loss, dice = self.loss.losses[self.loss.names[0]](preds_soft, gt_onehot)
                else:
                    loss = self.loss.losses[self.loss.names[0]](preds_soft, gt_onehot).cuda(self.config.device)
                    dice = compute_per_channel_dice(preds_soft, gt_onehot)
                try:
                    losses.append(loss.item())
                except Exception as e:
                    print(e)
                    print(loss)
                    print(loss.item())
                    sys.exit()

            self.scaler.scale(loss).backward()
            # loss.backward()
            self.scaler.step(self.optimizer)
            # self.optimizer.step()
            self.scaler.update()

            # hard_preds = torch.argmax(preds_soft, dim=1)
            # self.evaluator.compute_metrics(hard_preds, gt)
            self.evaluator.add_dice(dice=dice)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_dice = self.evaluator.mean_metric(phase='Train')

        self.metrics['Train'] = {
            'dice': epoch_dice,
        }

        wandb.log({
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Dice': epoch_dice,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        })

        return epoch_train_loss, epoch_dice

    def test(self, phase):

        self.model.eval()

        with torch.no_grad():
            torch.cuda.empty_cache()
            self.evaluator.reset_eval()
            losses = []

            if phase == 'Test':
                data_loader = self.test_loader
            elif phase == 'Validation':
                data_loader = self.val_loader

            for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'{phase} epoch {str(self.epoch)}'):
                images, gt = self.extract_data_from_feature(d)

                output = self.model(images)
                output_soft = F.softmax(output, dim=1)
                # 이미 여기서 thread: [23,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size
                gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=self.num_classes)
                gt_onehot = gt_onehot.unsqueeze(0)
                gt_onehot = torch.movedim(gt_onehot, -1, 1)
                assert output.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {output.ndim} vs {gt.ndim}'

                if self.loss.names[0] == 'Dice3DLoss':
                    loss, dice = self.loss.losses[self.loss.names[0]](output_soft, gt_onehot)
                else:
                    dice = compute_per_channel_dice(output_soft, gt_onehot)
                    loss = self.loss.losses[self.loss.names[0]](output_soft, gt_onehot).cuda(self.config.device)
                losses.append(loss.item())
                # self.evaluator.compute_metrics(output, gt)
                self.evaluator.add_dice(dice=dice)

            epoch_loss = sum(losses) / len(losses)
            epoch_dice = self.evaluator.mean_metric(phase=phase)

            wandb.log({
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss,
                f'{phase}/Dice': epoch_dice,
            })

            return epoch_dice
        

    def inference(self, output_path, phase='Test'):
        self.model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()

            if phase == 'Test':
                dataset = self.test_dataset
            elif phase == 'Validation':
                dataset = self.val_dataset
            elif phase == 'Train':
                dataset = self.train_dataset

            for i, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):
                os.makedirs(output_path, exist_ok=True)
                file_path = os.path.join(output_path, subject.patient+'_pred.seg.nrrd')
                # final_shape = subject.data.data[0].shape
                if os.path.exists(file_path) and False:
                    logging.info(f'skipping {subject.patient}...')
                    continue

                sampler = tio.inference.GridSampler(
                    subject,
                    self.config.data_loader.patch_shape,
                    0
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler, overlap_mode='hann')
            

                for j, patch in enumerate(loader):
                    images = patch['ct'][tio.DATA].float().cuda(self.config.device) 
                    preds = self.model(images)
                    aggregator.add_batch(preds, patch[tio.LOCATION])
                   
                output = aggregator.get_output_tensor()
                output_soft = F.softmax(output, dim=1)
                hard_output = torch.argmax(output_soft, dim=0)
                # hard_output = hard_output.squeeze(0)
                output = hard_output.detach().cpu().numpy()
                nrrd.write(file_path, np.uint8(output))
                logging.info(f'patient {subject.patient} completed, {file_path}.')