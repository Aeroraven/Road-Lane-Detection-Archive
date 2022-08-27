import os
import os.path
import pdb
import sys
from typing import Tuple

import jpeg4py
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.augmentation import *
from utils.system import file_readlines

from PIL import Image

class CULaneDatasetMK4(torch.utils.data.Dataset):
    """
    The improved version of CULaneDatasetMK3
    Part of code is referenced from cfzd
    https://github.com/cfzd/Ultra-Fast-Lane-Detection/main/data/dataset.py#L10
    """

    def __init__(self,
                 image_path: str,
                 seg_mask_path: str,
                 sub_directories = None,
                 c: int = 4,
                 w: int = 200,
                 h: int = 59,
                 h_min: int = 30,
                 wm: int = 10,
                 ih: int = 480,
                 iw: int = 800,
                 mask_downscale: int = 8,
                 augmentation: callable = get_tu_simple_augmentation,
                 preprocessing: callable = None,
                 is_test: bool = True,
                 train_split=0.8,
                 sanity_check: bool = False,
                 test_only:bool = False):
        super(CULaneDatasetMK4, self).__init__()
        if sub_directories is None:
            sub_directories = ['driver_161_90frame']
        self.files = []
        self.is_test = is_test
        self.mask_downscale = mask_downscale
        self.mask_idx = {}
        self.line_idx = {}
        self.c = c
        self.w = w
        self.h = h
        self.hm = h_min
        self.wm = wm
        self.ih = ih
        self.iw = iw
        self.test_only = test_only
        self.row_anchor_kf = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
        self.row_anchor = [int(i / 288 * ih) for i in self.row_anchor_kf]
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.fixed_aug = self.get_default_augment()
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
                            self.line_idx[i + "/" + r] = np.zeros(
                                (1))  # self.parse_line(i + "/" + r[:-4] + ".lines.txt")
                            if test_only:
                                self.files.append(i + "/" + r)
                                continue
                            if os.path.exists(seg_mask_path + "/" + ip + "/" + r[:-4] + ".png"):
                                self.mask_idx[i + "/" + r] = seg_mask_path + "/" + ip + "/" + r[:-4] + ".png"
                                if not sanity_check:
                                    self.files.append(i + "/" + r)
                                    valid += 1
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
                                        self.files.append(i + "/" + r)
                                        valid += 1
                            else:
                                print(seg_mask_path + "/" + ip + "/" + r[:-4] + ".png")
                                raise Exception(seg_mask_path + "/" + ip + "/" + r[:-4] + ".png")

                    f.set_postfix(valid=valid, anchor_corrupted=anchor_corrupted, mask_corrupted=mask_corrupted,
                                  corrupted=corrupted)
                    f.update(1)
            random.seed(3407)
            random.shuffle(self.files)
            if not self.is_test:
                self.files = self.files[:int(len(self.files) * train_split)]
            else:
                self.files = self.files[int(len(self.files) * train_split):]

    def get_default_augment(self):
        return ULCustomCompose([
            ULRandomRotate(6),
            ULRandomUDOffsetLABEL(100),
            ULRandomLROffsetLABEL(200)
        ])

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

    def albu_resize_mask(self):
        train_transform = albu.Compose([albu.Resize(self.ih // self.mask_downscale,
                                                    self.iw // self.mask_downscale, interpolation=cv2.INTER_NEAREST)])
        return train_transform

    def __getitem__(self, item):
        fn = self.files[item]
        return self.__getitem_impl__(fn)

    def __getitem_impl__(self, fn, sanity_test=False):
        if self.test_only:
            image = Image.open(fn).resize((1640, 590))
            image = np.array(image)
            resize_sample = self.albu_resize()(image=image)
            image = resize_sample['image']
            sample = self.preprocessing(image=image)
            image = sample['image']
            return image,fn
        image = Image.open(fn).resize((1640,590))
        mask = Image.open(self.mask_idx[fn])

        # Perform default augmentation
        if not self.is_test:
            image, mask = self.fixed_aug(image, mask)
            pass
        
        # Refinement starts here
        lane_pts = self.get_index(mask)
        image = np.array(image)
        mask = np.array(mask)
        mask_org = mask.copy()
        cls_label = self.grid_pts(lane_pts, self.w, mask.shape[1])
        anchor = cls_label
        anchor = np.transpose(anchor,(1,0))

        # Refinement ends
        resize_sample = self.albu_resize()(image=image)
        resize_mask = self.albu_resize_mask()(image=mask)
        image = resize_sample['image']
        mask = resize_mask['image']
        # mask = np.transpose(mask, (2, 0, 1))[0]
        if self.augmentation and not sanity_test and not self.is_test:
            sample = self.augmentation()(image=image)
            image = sample['image']
        if self.preprocessing and not sanity_test:
            sample = self.preprocessing(image=image)
            image = sample['image']
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

    def grid_pts(self, pts, num_cols, w):
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def get_index(self, label):
        def find_start_pos(row_sample, start_line_x):
            lt, rt = 0, len(row_sample) - 1
            while True:
                mid = int((lt + rt) / 2)
                if rt - lt == 1:
                    return rt
                if row_sample[mid] < start_line_x:
                    lt = mid
                if row_sample[mid] > start_line_x:
                    rt = mid
                if row_sample[mid] == start_line_x:
                    return mid

        w, h = label.size

        # assert h != self.ih
        scale_f = lambda x: int((x * 1.0 / self.ih) * h)
        sample_tmp = list(map(scale_f, self.row_anchor))
        all_idx = np.zeros((self.c, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.c + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        all_idx_cp = all_idx.copy()
        for i in range(self.c):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue
            if len(valid_idx) < 6:
                continue
            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp


if __name__ == "__main__":
    dataset = CULaneDatasetMK4(r"E:\Nf\tr2", r"E:\Nf\ts")
    print(dataset[0])
