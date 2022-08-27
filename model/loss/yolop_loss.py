import numpy as np
import torch
import torch.nn as nn
import torchmetrics


class YOLOPFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(YOLOPFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOPIouCachedMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, gt):
        return YOLOPIoULoss.value


class YOLOPIoULossV2(nn.Module):
    value = 0

    def __init__(self):
        super(YOLOPIoULossV2, self).__init__()
        self.n_classes = 2

    def forward(self, pred, target):
        N = len(pred)
        # pred = torch.softmax(pred, dim=1)
        inter = pred * target
        inter = inter.view(N, self.n_classes, -1).sum(2)
        union = pred + target - (pred * target)
        union = union.view(N, self.n_classes, -1).sum(2)
        loss = inter / (union + 1e-16)
        YOLOPIoULoss.value = loss.mean()
        return YOLOPIoULoss.value


class YOLOPIoULoss(nn.Module):
    value = 0

    def __init__(self) -> None:
        super().__init__()
        self.iou = torchmetrics.JaccardIndex(2).to("cuda")

    def forward(self, pred, target):
        YOLOPIoULoss.value = self.iou(pred, target)
        return YOLOPIoULoss.value

    def to(self, device):
        self.iou.to(device)


class YOLOPIoULossV1(nn.Module):
    def __init__(self):
        super(YOLOPIoULossV1, self).__init__()
        self.confusionMatrix = np.zeros((2,) * 2)
        self.numClass = 2

    def gen_confusion_matrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def iou(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def forward(self, pr, gt):
        self.confusionMatrix = self.gen_confusion_matrix(pr.cpu(), gt.cpu())
        return self.iou()


class YOLOPLaneLoss(nn.Module):
    def __init__(self, pads, device):
        super(YOLOPLaneLoss, self).__init__()
        self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.2]).to(device))
        self.pads = pads
        self.iou = YOLOPIoULoss()
        pad_w, pad_h = self.pads
        self.pad_w = int(pad_w)
        self.pad_h = int(pad_h)
        self.device = device

    def forward(self, pr, gt):
        lane_line_seg_predicts = pr.view(-1)
        lane_line_seg_targets = gt.view(-1)
        lseg_ll = self.bce_seg(lane_line_seg_predicts, lane_line_seg_targets)
        nb, _, height, width = gt.shape
        lane_line_pred = pr[:, :, self.pad_h:height - self.pad_h, self.pad_w:width - self.pad_w]
        lane_line_gt = gt[:, 1, self.pad_h:height - self.pad_h, self.pad_w:width - self.pad_w].type(torch.int64).to(
            self.device)
        liou_ll = 1 - self.iou(lane_line_pred, lane_line_gt)
        return lseg_ll + liou_ll

    def to(self, device):
        self.bce_seg.to(device)
        self.iou.to(device)
