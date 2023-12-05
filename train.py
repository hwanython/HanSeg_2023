import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import pathlib
import builtins
import socket
import random
import time
import json

import numpy as np
import torch
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from hashlib import shake_256
from munch import Munch, munchify, unmunchify
from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
import wandb

from utils.TaskFactory import *
from utils.AugmentFactory import *
from utils.DataPreprocFactory import *
from utils.ArrayIOFactory import DicomIO, MedicalIO, NumpyIO

# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5) # output len: 2*5=10
    return h.upper()

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("-c", "--config", default="configs/config-AttUNet3D-asahi.yaml",
    arg_parser.add_argument("-c", "--config", default="configs/config-AttUNet3D-osstem.yaml",                            
                            help="the config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", default=False, action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit
    
    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)



    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)
    
    # set the title name with timehash
    config.title = f'{config.title}_{timehash()}'

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    logging.info(f'setup wandb log ...')
    wandb.init(
        project="On3_dev_nerve",
        # entity=config.title,
        config=unmunchify(config)
    )
    # get run name
    run_name = wandb.run.name 

    # set run name
    wandb.run.name = config.title
    wandb.run.save()


    # check if augmentations is set and file exists
    logging.info(f'loading augmentations')
    if config.augmentations is None:
        aug = []
    else:
        aug = config.augmentations
    config.augmentations = AugFactory(aug).get_transform()



    logging.info(f'Instantiation of the experiment')
    # pdb.set_trace()
    experiment = TaskFactory(config, args.debug).get()
    logging.info(f'experiment title: {experiment.config.title}')

    project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.experiment.name, 'train', experiment.config.title)
    os.makedirs(project_dir_title, exist_ok=True)
    logging.info(f'project directory: {project_dir_title}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(project_dir_title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(experiment.config.data_loader.dataset):
        logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
        raise SystemExit

    # pre-calculate the checkpoints path
    checkpoints_path = path.join(project_dir_title, 'checkpoints')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
        raise SystemExit

    best_val = float('-inf')
    best_test = {
        'value': float('-inf'),
        'epoch': -1
    }


    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs+1):
            epoch_train_loss, epoch_iou, epoch_dice = experiment.train()
            logging.info(f'Epoch {epoch} Train IoU: {epoch_iou}')
            logging.info(f'Epoch {epoch} Train Dice: {epoch_dice}')
            logging.info(f'Epoch {epoch} Train Loss: {epoch_train_loss}')

            val_iou, val_dice = experiment.test(phase="Validation")
            logging.info(f'Epoch {epoch} Val IoU: {val_iou}')
            logging.info(f'Epoch {epoch} Val Dice: {val_dice}')
            if val_iou < 1e-05 and experiment.epoch > 15:
                logging.warning('WARNING: drop in performances detected.')

            optim_name = experiment.optimizer.name
            sched_name = experiment.scheduler.name

            if experiment.scheduler is not None:
                if optim_name == 'SGD' and sched_name == 'Plateau':
                    experiment.scheduler.step(val_iou)
                elif sched_name == 'SGDR':
                    experiment.scheduler.step()
                else:
                    experiment.scheduler.step(epoch)

            if epoch % 3 == 0:
                test_iou, test_dice = experiment.test(phase="Test")
                logging.info(f'Epoch {epoch} Test IoU: {test_iou}')
                logging.info(f'Epoch {epoch} Test Dice: {test_dice}')

                if test_iou > best_test['value']:
                    best_test['value'] = test_iou
                    best_test['epoch'] = epoch

            experiment.save('last.pth')

            if val_iou > best_val:
                best_val = val_iou
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best test IoU found: {best_test['value']} at epoch: {best_test['epoch']}
                ''')
        
    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')

    # Do the inference
    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        output_path = r'experiments/test'
        output_path = os.path.join(output_path, config.title)
        os.makedirs(output_path, exist_ok=True)
        # Copy config file to project_dir, to be able to reproduce the experiment
        copy_config_path = os.path.join(output_path, 'config.yaml')
        shutil.copy(args.config, copy_config_path)
        if config.trainer.inference_zslide:
            experiment.inference(output_path=output_path, phase="inference", zslide=True)
        else:
            experiment.inference(output_path=output_path, phase="inference")
        