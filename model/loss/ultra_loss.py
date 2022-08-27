import torch
import torch.nn as nn
import torch.nn.functional as torch_func
import torchmetrics
import numpy as np

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = torch_func.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = torch_func.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class UltraLaneAccuracy(nn.Module):
    def __init__(self, device="cuda"):
        super(UltraLaneAccuracy, self).__init__()
        self.accm = torchmetrics.Accuracy().to(device)

    def forward(self, pr, gt):
        pr = torch.argmax(pr, dim=1)
        return 1-self.accm(pr, gt)
