from statistics import mean
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from skimage import metrics
import os
import pandas as pd
import zipfile
from monai.metrics import DiceMetric

class Eval:
    def __init__(self, classes):
        self.eps = 1e-06
        self.classes=classes
        self.dice_list = []

    def reset_eval(self):
        self.dice_list.clear()

    def get_dice(self,  hard_preds, gt):
        # those are B 1 H W D
        hard_preds_onehot = torch.nn.functional.one_hot(hard_preds, self.classes).permute(0, 4, 1, 2, 3)
        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=self.classes)
        gt_onehot = gt_onehot.unsqueeze(0)
        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        cal_dice = DiceMetric(include_background=False, reduction="mean")(hard_preds_onehot, gt_onehot)
        metric = cal_dice.aggregate()


        return 
    
    def mean_metric(self, phase):
        dice = 0 if len(self.dice_list) == 0 else mean(self.dice_list)
        self.reset_eval()
        return dice
