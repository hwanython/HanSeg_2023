import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from monai.losses import DiceCELoss

class LossFactory:
    def __init__(self, names, classes, weights=None):
        self.names = names
        if not isinstance(self.names, list):
            self.names = [self.names]

        print(f'Losses used: {self.names}')
        self.classes = classes
        self.weights = weights
        self.losses = {}
        for name in self.names:
            loss = self.get_loss(name)
            self.losses[name] = loss

    def get_loss(self, name):
        if name == 'JaccardLoss':
            loss_fn = JaccardLoss(weight=self.weights)
        elif name == 'Dice3DLoss':
            loss_fn = Dice3DLoss(weight=self.weights)
        elif name == 'DiceCELoss':
            loss_fn = DiceCELoss(self.classes, to_onehot_y=False, softmax=True)
        elif name == 'DiceLoss':
            loss_fn = DiceLoss(self.classes)
        elif name == 'BoundaryLoss':
            loss_fn = BoundaryLoss(self.classes)
        else:
            raise Exception(f"Loss function {name} can't be found.")
        return loss_fn

class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_volume=False, apply_sigmoid=True,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.weight = weight
        self.per_volume = per_volume
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, pred, gt):
        assert pred.shape[1] == 1, 'this loss works with a binary prediction'
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        batch_size = pred.size()[0]
        eps = 1e-6
        if not self.per_volume:
            batch_size = 1
        dice_gt = gt.contiguous().view(batch_size, -1).float()
        dice_pred = pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_pred * dice_gt, dim=1)
        union = torch.sum(dice_pred + dice_gt, dim=1) - intersection
        loss = 1 - (intersection + eps) / (union + eps)
        return loss
    
class DiceLoss(nn.Module):
    # TODO: Check about partition_weights, see original code
    # what i didn't understand is that for dice loss, partition_weights gets
    # multiplied inside the forward and also in the factory_loss function
    # I think that this is wrong, and removed it from the forward
    def __init__(self, classes):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes

    def forward(self, pred, gt):
        # included = [v for k, v in self.classes.items() if k not in ['UNLABELED']]
        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=self.classes)
        # if gt.shape[0] = 1:  # we need to add a further axis after the previous squeeze()
        gt_onehot = gt_onehot.unsqueeze(0)
        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        input_soft = F.softmax(pred, dim=1)
        dims = (2, 3, 4)

        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return 1. - torch.mean(dice_score)
    




def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Ref: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # return 2 * (intersect / denominator.clamp(min=epsilon))
    return 2 * ((intersect + epsilon) / (denominator + epsilon))


class Dice3DLoss(torch.nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.

    Args:
         input (torch.Tensor): NxCxDxHxW input tensor
         target (torch.Tensor): NxCxDxHxW target tensor
    """

    def __init__(self, weight=None):
        super(Dice3DLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.weight = weight

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = compute_per_channel_dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice), per_channel_dice


class BoundaryLoss(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def DiceCoeff(self, pred, target, smooth=1e-7):
        inter = (pred * target).sum()
        return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

    def DiceLoss(self, pred, target, smooth=1e-7):
        return torch.sum(1 - self.DiceCoeff(pred, target, smooth))

    def extract_surface(self, volume):
        # Pad the volume with zeros on all sides
        padded_volume = torch.nn.functional.pad(volume, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        # Compute the gradient along all three dimensions
        dz = padded_volume[1:-1, 1:-1, 2:] - padded_volume[1:-1, 1:-1, :-2]
        dy = padded_volume[1:-1, 2:, 1:-1] - padded_volume[1:-1, :-2, 1:-1]
        dx = padded_volume[2:, 1:-1, 1:-1] - padded_volume[:-2, 1:-1, 1:-1]

        # Compute the magnitude of the gradient vector
        mag = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        mag[mag > 0] = 1
        return mag

    def forward(self, preds, targets):
        # sigmoid probability map
        preds_ = torch.sigmoid(preds)
        p_set = self.extract_surface(preds_[0, 0, ...]).cuda()
        t_set = self.extract_surface(targets[0, 0, ...]).cuda()
        loss = self.DiceLoss(p_set, t_set)

        return loss

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None, apply_sigmoid=True):
        super().__init__()
        self.weights = weights
        self.apply_sigmoid = apply_sigmoid
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pred, gt):
        pred = self.sigmoid(pred)
        return self.loss_fn(pred, gt)


class BoundaryLoss(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def DiceCoeff(self, pred, target, smooth=1e-7):
        inter = (pred * target).sum()
        return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

    def DiceLoss(self, pred, target, smooth=1e-7):
        return torch.sum(1 - self.DiceCoeff(pred, target, smooth))

    def extract_surface(self, volume):
        # Pad the volume with zeros on all sides
        padded_volume = torch.nn.functional.pad(volume, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        # Compute the gradient along all three dimensions
        dz = padded_volume[1:-1, 1:-1, 2:] - padded_volume[1:-1, 1:-1, :-2]
        dy = padded_volume[1:-1, 2:, 1:-1] - padded_volume[1:-1, :-2, 1:-1]
        dx = padded_volume[2:, 1:-1, 1:-1] - padded_volume[:-2, 1:-1, 1:-1]

        # Compute the magnitude of the gradient vector
        mag = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        mag[mag > 0] = 1
        return mag

    def forward(self, preds, targets):
        # sigmoid probability map
        preds_ = torch.sigmoid(preds)
        p_set = self.extract_surface(preds_[0, 0, ...]).cuda()
        t_set = self.extract_surface(targets[0, 0, ...]).cuda()
        loss = self.DiceLoss(p_set, t_set)

        return loss