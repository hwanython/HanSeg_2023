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
from torch.nn.parallel import DistributedDataParallel
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HaN import HaN
from libs.losses.LossFactory import LossFactory
from libs.models.ModelFactory import ModelFactory
from libs.optimizers.OptimizerFactory import OptimizerFactory
from libs.schedulers.SchedulerFactory import SchedulerFactory
from utils.AugmentFactory import *
from utils.EvalFactory import Eval as Evaluator
from utils.DataPreprocFactory import *

eps = 1e-10


class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json'
        if self.debug:
            filename = 'splits_small.json'

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1

        self.model = ModelFactory(model_name, num_classes, in_ch).get().cuda()

        
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
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = HaN(
            root=self.config.data_loader.dataset,
            filename=filename,
            splits='train',
            transform=tio.Compose([
                # tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                self.config.augmentations,
                tio.Resize(self.config.data_loader.resize_shape),
                
            ]),
            sampler=self.config.data_loader.sampler_type
        )
        self.val_dataset = HaN(
            root=self.config.data_loader.dataset,
            transform=tio.Compose([
                tio.Resize(self.config.data_loader.resize_shape)]),
            filename=filename,
            splits='val',
        )
        self.test_dataset = HaN(
            root=self.config.data_loader.dataset,
            transform=tio.Compose([
                tio.Resize(self.config.data_loader.resize_shape)]),
            filename=filename,
            splits='test',
        )

        self.inference_dataset = HaN(
            root=self.config.data_loader.dataset,
            filename=filename,
            splits='test',
        )

        # self.test_aggregator = self.train_dataset.get_aggregator(self.config.data_loader)
        # self.synthetic_aggregator = self.synthetic_dataset.get_aggregator(self.config.data_loader)

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
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
        volume = feature['data'][tio.DATA].float().cuda()
        gt = feature['dense'][tio.DATA].float().cuda()
        # volume = volume/255. # normalization

        if self.config.experiment.name == 'Generation':
            sparse = feature['sparse'][tio.DATA].float().cuda()
            images = torch.cat([volume, sparse], dim=1)
        else:
            images = volume
        
        return images, gt

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader
        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt = self.extract_data_from_feature(d)

            self.optimizer.zero_grad()
            preds = self.model(images)

            assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
            loss = self.loss.losses[self.loss.names[0]](preds, gt).cuda()
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            preds = (preds > 0.5).squeeze().detach()

            gt = gt.squeeze()
            self.evaluator.compute_metrics(preds, gt)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')

        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }

        wandb.log({
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Dice': epoch_dice,
            f'Train/IoU': epoch_iou,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        })

        return epoch_train_loss, epoch_iou, epoch_dice

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
                assert output.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {output.ndim} vs {gt.ndim}'

                loss = self.loss.losses[self.loss.names[0]](output.unsqueeze(0), gt.unsqueeze(0)).cuda()
                losses.append(loss.item())

                output = output.squeeze(0)
                output = (output > 0.5)

                self.evaluator.compute_metrics(output, gt)

            epoch_loss = sum(losses) / len(losses)
            epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)

            wandb.log({
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss,
                f'{phase}/Dice': epoch_dice,
                f'{phase}/IoU': epoch_iou,
            })

            return epoch_iou, epoch_dice
        

    def inference(self, output_path, phase='inference', zslide=False):

        self.model.eval()

        with torch.no_grad():
            torch.cuda.empty_cache()

            if phase == 'Test':
                dataset = self.test_dataset
                data_loader = self.test_loader
            elif phase == 'Validation':
                dataset = self.val_dataset
                data_loader = self.val_loader
            elif phase == 'Train':
                dataset = self.train_dataset
                data_loader = self.train_loader
            elif phase == 'inference':
                dataset = self.inference_dataset
                data_loader = self.inference_dataset.get_loader(self.config.data_loader)


            for i, (ds, dl) in tqdm(enumerate(zip(dataset._subjects, data_loader)), total=len(data_loader), desc=f'{phase} epoch {str(self.epoch)}'):
                volume = dl['data'][tio.DATA].float().cuda()
                input_resize_size = self.config.data_loader.resize_shape

                if zslide:
                    z_ratio = 0.3
                    z1 = volume.shape[-3] * z_ratio
                    _crop1 = tio.Crop((0,int(z1),0,0,0,0))
                    # z2 = volume.shape[-3] * (1 - z_ratio)
                    _crop2 = tio.Crop((int(z1),0,0,0,0,0))
                    volume = volume.squeeze(0)
                    volume1 = _crop1(volume)
                    volume2 = _crop2(volume)
                    _volume = [volume1, volume2]
                    _output = []

                    target_size = ds['data'][tio.DATA].squeeze(0).detach().cpu().numpy().shape
                    final_out = np.zeros(target_size, dtype=np.uint8)
                   

                    for i, v in enumerate(_volume):
                        v = v.unsqueeze(0)
                        v = F.interpolate(v, size= input_resize_size, mode='trilinear')
                        o = self.model(v)
                        o = o.squeeze(0)
                        o = o.detach().cpu().numpy()
                        resize = tio.Resize((_volume[i][0].shape[0], _volume[i][0].shape[1], _volume[i][0].shape[2]))
                        o = resize(o)
                        o = o.squeeze(0)
                        o = (o > 0.5)
                        _output.append(o)

                    final_out[:target_size[0]-int(z1),:, :] += _output[0].astype(np.uint8)
                    final_out[int(z1):,:, :] += _output[1].astype(np.uint8)
                    output = final_out
                    output[output>0] = 1
                else:
                    volume = F.interpolate(volume, size=input_resize_size, mode='trilinear')
                    output = self.model(volume)
                    output = output.squeeze(0)
                    target_size = ds['data'][tio.DATA].squeeze(0).detach().cpu().numpy().shape
                    resize = tio.Resize(target_size)
                    output = output.detach().cpu().numpy()
                    output = resize(output)
                    output = output.squeeze(0)
                    output = (output > 0.5)

                os.makedirs(output_path, exist_ok=True)
                file_path = os.path.join(output_path, ds.patient+'_pred.seg.nrrd')
                nrrd.write(file_path, np.uint8(output))
                logging.info(f'patient {ds.patient} completed, {file_path}.')
