import math
import random

import albumentations as albu
import cv2
import numpy as np
from PIL import Image


class ULCustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class ULRandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size

        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label


class ULRandomLROffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


class ULRandomUDOffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)


def get_tu_simple_augmentation():
    lst = [
        albu.ImageCompression(always_apply=False, p=0.5, quality_lower=80, quality_upper=100, compression_type=0),
        albu.GaussNoise(always_apply=False, p=0.4, var_limit=(10.0, 180.9199981689453)),
        albu.CoarseDropout(always_apply=False, p=0.3, max_holes=64, max_height=8, max_width=8, min_holes=16,
                           min_height=8,
                           min_width=8),
        albu.ChannelShuffle(always_apply=False, p=0.1),
        albu.RGBShift(always_apply=False, p=0.1,
                      r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
        albu.CLAHE(always_apply=False, p=0.2, clip_limit=(1, 4), tile_grid_size=(8, 8))
    ]
    transform = albu.Compose(lst)
    return transform


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(combination, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)

    return img, gray


def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    img, gray = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    # print(img.shape)

    combination = (img, gray)
    return combination, ratio, (dw, dh)
