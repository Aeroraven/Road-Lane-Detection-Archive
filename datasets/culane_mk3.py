import os
import os.path
import random
import sys
from typing import Tuple

import cv2
import jpeg4py
import numpy as np
import torch.utils.data
import albumentations as albu
from tqdm import tqdm

from utils.augmentation import get_tu_simple_augmentation
from utils.system import file_readlines


class CULaneDatasetMK3(torch.utils.data.Dataset):
    def __init__(self,
                 image_path: str,
                 seg_mask_path: str,
                 sub_directories = None,
                 c: int = 4,
                 w: int = 170,
                 h: int = 59,
                 h_min: int = 30,
                 wm: int = 10,
                 ih: int = 480,
                 iw: int = 800,
                 augmentation: callable = get_tu_simple_augmentation,
                 preprocessing: callable = None,
                 is_test: bool = True,
                 train_split=0.8,
                 sanity_check: bool = False):
        super(CULaneDatasetMK3, self).__init__()
        if sub_directories is None:
            sub_directories = ['driver_161_90frame']
        self.files = []
        self.is_test = is_test
        self.mask_idx = {}
        self.line_idx = {}
        self.c = c
        self.w = w
        self.h = h
        self.hm = h_min
        self.wm = wm
        self.ih = ih
        self.iw = iw
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        for sd in sub_directories:
            lst = list(os.walk(image_path))
            valid = 0
            corrupted = 0
            anchor_corrupted = 0
            mask_corrupted = 0
            with tqdm(total=len(lst), desc="Indexing Dataset", ascii=True, file=sys.stdout) as f:
                for i, j, k in lst:
                    for r in k:
                        if r.endswith(".jpg"):
                            ip = i.replace(image_path, "")
                            self.line_idx[i + "/" + r] = np.zeros((1))# self.parse_line(i + "/" + r[:-4] + ".lines.txt")
                            if os.path.exists(seg_mask_path + "/" + ip + "/" + r[:-4] + ".png"):

                                self.mask_idx[i + "/" + r] = seg_mask_path + "/" + ip + "/" + r[:-4] + ".png"
                                valid += 1
                                if not sanity_check:
                                    self.files.append(i + "/" + r)
                                else:
                                    image, fp = self.__getitem_impl__(i + "/" + r, True)
                                    fr = np.min(fp[1])
                                    fw = np.min(fp[0])
                                    if fr == 0 and fw == 0:
                                        corrupted += 1
                                    elif fr == 0:
                                        anchor_corrupted += 1
                                    elif fw == 0:
                                        mask_corrupted += 1
                            else:
                                print(seg_mask_path + "/" + ip + "/" + r[:-4] + ".png")
                                raise

                    f.set_postfix(valid=valid, anchor_corrupted=anchor_corrupted, mask_corrupted=mask_corrupted,
                                  corrupted=corrupted)
                    f.update(1)
            random.seed(3407)
            random.shuffle(self.files)
            if not self.is_test:
                self.files = self.files[:int(len(self.files) * train_split)]
            else:
                self.files = self.files[int(len(self.files) * train_split):]

    def parse_line(self, file):
        grids = [250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590]
        dict_y = dict()
        for i in range(len(grids)):
            dict_y[grids[i]] = i
        s_map = np.ones((self.c, 18)) * self.w
        fr = file_readlines(file)
        for i in range(len(fr)):
            fr[i] = fr[i].replace('\n', '').rstrip()
            kp = fr[i].split(' ')
            for j in range(0, len(kp), 2):
                xpos = int(float(kp[j])) // self.wm
                if xpos < 0:
                    continue
                if xpos >= self.w:
                    continue
                if int(kp[j + 1]) in dict_y:
                    s_map[i, dict_y[int(kp[j + 1])]] = xpos

        return s_map

    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.ih, self.iw, interpolation=cv2.INTER_NEAREST)])
        return train_transform

    def __getitem__(self, item):
        fn = self.files[item]
        return self.__getitem_impl__(fn)

    def __getitem_impl__(self, fn, sanity_test=False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        image = jpeg4py.JPEG(fn).decode()
        mask = cv2.imread(self.mask_idx[fn])
        if self.line_idx[fn].shape[0] == 1:
            self.line_idx[fn] = self.parse_line(fn[:-4] + ".lines.txt")
        anchor: np.ndarray = self.line_idx[fn]
        resize_sample = self.albu_resize()(image=image, mask=mask)
        image = resize_sample['image']
        mask = resize_sample['mask']
        mask = np.transpose(mask, (2, 0, 1))[0]
        if self.augmentation and not sanity_test and not self.is_test:
            sample = self.augmentation()(image=image)
            image = sample['image']
        if self.preprocessing and not sanity_test:
            sample = self.preprocessing(image=image)
            image = sample['image']
        if np.min(anchor) != self.w and np.max(mask) == 0:
            raise
        if type(anchor) is str:
            raise
        if np.max(anchor) > self.w:
            raise
        if np.min(anchor) < 0:
            raise Exception("Minval", np.min(anchor))
        if not self.is_test:
            image_r, mask_r, anchor_r = self.data_aug(image, mask, anchor)
        else:
            return image, (mask.astype("int64"), anchor.astype("int64"))
        return image_r, (mask_r.astype("int64"), anchor_r.astype("int64"))

    def data_aug(self, image, mask, anchor):
        # LR Flip
        if random.random() < -1:
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
            anchor = (self.w - 1) - anchor
            anchor[anchor < 0] = self.w
        return image.copy(), mask.copy(), anchor.copy()

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    dataset = CULaneDatasetMK3(r"E:\新建文件夹 (4)\driver_161_90frame",
                               r"E:\Nf\laneseg_label_w16\laneseg_label_w16")
    for i in tqdm(range(500)):
        sf = dataset[i]
