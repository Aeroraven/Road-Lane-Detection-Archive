import torch
import torch.nn as nn
from segmentation_models_pytorch.utils import functional as smpf
import numpy as np
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.base.modules import Activation
from torch.autograd import Variable


def make_one_hot(input_var, num_classes):
    shape = np.array(input_var.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input_var.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(base.Loss):
    def __init__(self, eps=1., beta=1., weight=1, activation=None, ignore_channels=None, single_channel=False,
                 selected_channel=0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.weight = weight
        self.single_channel = single_channel
        self.selected_channel = selected_channel

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        if self.single_channel:
            y_pr = y_pr[:, self.selected_channel]
            y_gt = y_gt[:, self.selected_channel]
        if y_pr.shape != y_gt.shape:
            raise Exception("Tensor shapes mismatch", " PR", y_pr.shape, " GT", y_gt.shape)
        return self.weight * (1 - smpf.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        ))


class DiceMetrics(base.Loss):
    def __init__(self, eps=1.,
                 beta=1.,
                 weight=1,
                 activation=None,
                 ignore_channels=None,
                 single_channel=False,
                 selected_channel=0,
                 expand=False,
                 classes=2,
                 device="cuda",
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.weight = weight
        self.single_channel = single_channel
        self.selected_channel = selected_channel
        self.expand = expand
        self.classes = classes
        self.device = device

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        if self.expand:
            y_gt = y_gt.detach().cpu().numpy()
            y_gts = [(y_gt == v) for v in range(self.classes)]
            y_gt = np.stack(y_gts, axis=-1).astype("float")
            y_gt = np.transpose(y_gt, (0, 3, 1, 2))
            y_gt = torch.tensor(y_gt).to(self.device)
        if self.single_channel:
            y_pr = y_pr[:, self.selected_channel]
            y_gt = y_gt[:, self.selected_channel]
        if y_pr.shape != y_gt.shape:
            raise Exception("Tensor shapes mismatch", " PR", y_pr.shape, " GT", y_gt.shape)
        imd = smpf.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
        return self.weight * imd


class DicePancLoss(base.Loss):
    def __init__(self, name=None):
        super().__init__(name)
        self.dice1 = DiceLoss(single_channel=True, selected_channel=0)
        self.dice2 = DiceLoss(single_channel=True, selected_channel=1)
        self.dice3 = DiceLoss(single_channel=True, selected_channel=2)

    def forward(self, y_pr, y_gt):
        d1 = self.dice1(y_pr, y_gt)
        d2 = self.dice2(y_pr, y_gt)
        d3 = self.dice3(y_pr, y_gt)
        return (d1 + d2 + d3) / 3.0

class DiceLaneLoss(base.Loss):
    def __init__(self, name=None):
        super().__init__(name)
        self.dice1 = DiceLoss(single_channel=True, selected_channel=0)
        self.dice2 = DiceLoss(single_channel=True, selected_channel=1)

    def forward(self, y_pr, y_gt):
        d1 = self.dice1(y_pr, y_gt)
        d2 = self.dice2(y_pr, y_gt)
        return d2*0.75+d1*0.25


class FocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, w=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.w = w
        return

    def forward(self, y_pr, y_gt):
        y_pr = y_pr - y_pr.max(dim=1, keepdim=True)[0]
        logits = torch.exp(y_pr)
        logits_sum = logits.sum(dim=1, keepdim=True)
        _, gt_index = y_gt.max(dim=1, keepdim=True)
        pt = Variable((logits / logits_sum).gather(dim=1, index=gt_index))
        pt = (1 - pt) ** self.gamma
        loss = -y_gt * (y_pr - torch.log(logits_sum))
        loss = self.alpha * pt * loss
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=[1, 2])
        return self.w * loss.mean()


class SwiftLaneCrossEntropyLoss(base.Loss):
    def __init__(self, c, h, w, w_mod, focal_gamma=1):
        super(SwiftLaneCrossEntropyLoss, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.wm = w_mod
        self.g = focal_gamma

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor):
        # Y_PR (BN,C,H,W+1)
        # Y_GT (BN,C,H)
        y_gt[y_gt < 0] = self.w * self.wm
        y_gt = y_gt / self.wm
        y_gt = torch.floor(y_gt)
        loss = 0
        for b in range(y_pr.shape[0]):
            for i in range(self.c):
                for j in range(self.h):
                    fv = y_pr[b, i, j, int(y_gt[b, i, j])]
                    loss += (-(1 - fv) ** self.g) * torch.log(fv)
        return loss


class SwiftLaneGridAccuracy(base.Loss):
    def __init__(self, c, h, w, w_mod, focal_gamma=1):
        super(SwiftLaneGridAccuracy, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.wm = w_mod
        self.g = focal_gamma

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor):
        y_gt[y_gt < 0] = self.w * self.wm
        y_gt = y_gt / self.wm
        y_gt = torch.floor(y_gt)
        y_pr = torch.argmax(y_pr, dim=3)
        cv = 0
        for b in range(y_pr.shape[0]):
            for i in range(self.c):
                for j in range(self.h):
                    if y_pr[b, i, j] == y_gt[b, i, j]:
                        cv += 1
        return cv / y_pr.shape[0] / y_pr.shape[1] / y_pr.shape[2]


class SwiftLaneRegressionEcuLoss(base.Loss):
    def __init__(self, c, h, w, w_mod, focal_gamma=1):
        super(SwiftLaneRegressionEcuLoss, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.wm = w_mod
        self.g = focal_gamma
        self.w1 = 1e4 # CrossEntropy-Exist
        self.w2 = 1 # Lane deviation
        self.w3 = 1e3 # Start & Termination Deviation

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor):
        # Y_PR (BN,3C+4+2C)
        # Y_GT (BN,C,H)
        y_gt[y_gt < 0] = self.w * self.wm
        y_gt = y_gt / self.wm
        y_gt = torch.floor(y_gt)
        loss = 0
        for b in range(y_pr.shape[0]):
            A = y_pr[b, 3 * self.c]
            B = y_pr[b, 3 * self.c + 1]
            C = y_pr[b, 3 * self.c + 2]
            P = y_pr[b, 3 * self.c + 3]
            D = y_pr[b, 0:self.c]
            AL = y_pr[b, self.c:2 * self.c]
            BE = y_pr[b, self.c * 2:self.c * 3]
            PR = y_pr[b, self.c * 3 + 4:]
            PR = torch.reshape(PR, (self.c, 2))
            for i in range(self.c):
                vis = 0
                st = 900
                ed = -900
                for j in range(self.h):
                    if y_gt[b, i, j] == self.w:
                        continue
                    vis += 1
                    st = min(st, j)
                    ed = max(ed, j)
                    pr_x = A * ((j - P) ** 2) + B / (j - P) + C + D[i] * (j - P)
                    if pr_x == torch.nan or pr_x == np.nan:
                        continue
                    loss += self.w2*abs(pr_x - y_gt[b, i, j])
                if vis == 0:
                    bg = -torch.log(PR[i, 0])
                    if bg == torch.nan or bg == np.nan:
                        raise Exception("NaN Loss")
                    elif bg == torch.inf or bg == np.inf:
                        import warnings
                        warnings.warn("Inf Log")
                    else:
                        loss += self.w1* bg
                else:
                    bg = -torch.log(PR[i, 1])
                    if bg == torch.nan or bg == np.nan:
                        raise Exception("NaN Loss")
                    elif bg == torch.inf or bg == np.inf:
                        import warnings
                        warnings.warn("Inf Log")
                    else:
                        loss += self.w1* bg
                    loss += self.w3 * abs(st - AL[i])
                    loss += self.w3 *abs(st - BE[i])
        return loss / y_pr.shape[0]


class SwiftLaneRegressionCE(base.Loss):
    def __init__(self, c, h, w, w_mod, focal_gamma=1):
        super(SwiftLaneRegressionCE, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.wm = w_mod
        self.g = focal_gamma

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor):
        # Y_PR (BN,3C+4+2C)
        # Y_GT (BN,C,H)
        y_gt[y_gt < 0] = self.w * self.wm
        y_gt = y_gt / self.wm
        y_gt = torch.floor(y_gt)
        loss = 0
        for b in range(y_pr.shape[0]):
            PR = y_pr[b, self.c * 3 + 4:]
            PR = torch.reshape(PR, (self.c, 2))
            for i in range(self.c):
                vis = 0
                for j in range(self.h):
                    if y_gt[b, i, j] == self.w:
                        continue
                    vis += 1
                    break
                if vis == 0:
                    bg = -torch.log(PR[i, 0])
                    if bg == torch.nan or bg == np.nan:
                        raise Exception("NaN Loss")
                    elif bg == torch.inf or bg == np.inf:
                        import warnings
                        warnings.warn("Inf Log")
                    else:
                        loss += bg
                else:
                    bg = -torch.log(PR[i, 1])
                    if bg == torch.nan or bg == np.nan:
                        raise Exception("NaN Loss")
                    elif bg == torch.inf or bg == np.inf:
                        import warnings
                        warnings.warn("Inf Log")
                    else:
                        loss += bg
        return loss / y_pr.shape[0]
