import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


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
        elif name == 'BCEWithLogitsViewLoss':
            loss_fn = BCEWithLogitsViewLoss(weight=self.weights)
        elif name == 'BCEDiceFocalLoss':
            loss_fn = BCEDiceFocalLoss(focal_param=2)
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
        included = [v for k, v in self.classes.items() if k not in ['UNLABELED']]
        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=len(self.classes))
        if gt.shape[0] == 1:  # we need to add a further axis after the previous squeeze()
            gt_onehot = gt_onehot.unsqueeze(0)

        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        input_soft = F.softmax(pred, dim=1)
        dims = (2, 3, 4)

        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return 1. - dice_score[:, included]

class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''

    def __init__(self, focal_param, weights=None, **kwargs):
        if weights is None:
            weights = [0.1, 1.0, 1.0]
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = BCEWithLogitsViewLoss(weight=None, size_average=True, **kwargs)
        self.dice = SoftDiceLoss(**kwargs)
        self.focal = FocalLoss(l=focal_param, **kwargs)
        self.weights = weights

    def forward(self, logits, labels, **_):
        return self.weights[0] * self.bce(logits, labels) + self.weights[1] * self.dice(logits, labels) + self.weights[
            2] * self.focal(logits.unsqueeze(1), labels.unsqueeze(1))


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **_):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels, **_):
        num = labels.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.

    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """

    def __init__(self, l=0.5, eps=1e-6, **_):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, labels, **_):
        labels = labels.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(labels * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - labels) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss


# ==== Custom ==== #
class BCEWithLogitsViewLoss(nn.BCEWithLogitsLoss):
    '''
    Silly wrapper of nn.BCEWithLogitsLoss because BCEWithLogitsLoss only takes a 1-D array
    '''

    def __init__(self, weight=None, size_average=True, **_):
        super().__init__(weight=weight, size_average=size_average)

    def forward(self, input_, target, **_):
        '''
        :param input_:
        :param target:
        :return:

        Simply passes along input.view(-1), target.view(-1)
        '''
        return super().forward(input_.view(-1), target.view(-1))


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