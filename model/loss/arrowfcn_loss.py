import torch.nn
import torchmetrics
import segmentation_models_pytorch as smp

from model.loss.legacy_loss import *


class ArrowFCNCELoss(torch.nn.Module):
    meter_value = 4.0
    cl1_dice = 0.0
    def __init__(self, device="cuda"):
        super(ArrowFCNCELoss, self).__init__()
        self.celoss = torch.nn.CrossEntropyLoss().to(device)
        self.meter = torchmetrics.JaccardIndex(2).to(device)
        self.dice_fore = smp.losses.DiceLoss("multiclass", from_logits=False, classes=[1])
        self.dice_back = smp.losses.DiceLoss("multiclass", from_logits=False, classes=[0])

    def forward(self, pr, gt):
        ArrowFCNCELoss.cl1_dice = self.dice_fore(pr, gt)
        return ArrowFCNCELoss.cl1_dice * 0.95 + self.dice_back(pr,gt) * 0.05


class ArrowFCNIoUMetric(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(ArrowFCNIoUMetric, self).__init__()
        self.meter = torchmetrics.JaccardIndex(2).to(device)

    def forward(self, pr, gt):
        return self.meter(pr,gt)


class ArrowFCNAccMetric(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(ArrowFCNAccMetric, self).__init__()
        self.meter = torchmetrics.Accuracy().to(device)

    def forward(self, pr, gt):
        return self.meter(pr,gt)

class ArrowFCNForeDiceMetric(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(ArrowFCNForeDiceMetric, self).__init__()
        self.meter = torchmetrics.Accuracy().to(device)

    def forward(self, pr, gt):
        return 1-ArrowFCNCELoss.cl1_dice
