import random

import torch
from torch.utils.data.dataset import Dataset

from datasets.apollo import ApolloDataset
from datasets.culane import CULaneDatasetMK2
from datasets.culane_mk3 import CULaneDatasetMK3
from datasets.culane_mk4 import CULaneDatasetMK4


class AlwenMixedDatasetV1(Dataset):
    def __init__(self,
                 cu_image_path: str,
                 cu_seg_mask_path: str,
                 ap_image_path: str,
                 ap_seg_mask_path: str,
                 preprocessing: callable = None,
                 is_test = False,
                 ih=800,
                 iw=480,
                 ):
        
        self.culane = CULaneDatasetMK4(cu_image_path, cu_seg_mask_path,
                                       preprocessing=preprocessing,
                                       is_test=is_test)
        self.apollo = ApolloDataset(ih, iw,
                                    arrow_path=ap_image_path,
                                    arrow_mask_path=ap_seg_mask_path,
                                    preprocessing=preprocessing,
                                    is_test=is_test)
        if is_test:
            self.sk = min(len(self.culane),len(self.apollo))
        else:
            self.sk = min(len(self.culane),len(self.apollo)) 
        self.total = self.sk * 2
        self.divide = self.sk
        self.culane_idx = list(range(len(self.culane)))
        self.apollo_idx = list(range(len(self.apollo)))

        self.culane_pool = random.choices(self.culane_idx, k=len(self.culane))
        self.apollo_pool = random.choices(self.apollo_idx, k=len(self.apollo))

    def __len__(self):
        return self.total

    def __getitem__(self, item):
        if item < self.divide:
            if len(self.culane_pool) == 0:
                self.culane_pool = random.choices(self.culane_idx, k=len(self.culane))
            xs, xt = self.culane[self.culane_pool.pop(random.randint(0, len(self.culane_pool) - 1))]
            return xs, xt
        else:
            if len(self.apollo_pool) == 0:
                self.apollo_pool = random.choices(self.apollo_idx, k=len(self.apollo))
            xs = self.apollo[self.apollo_pool.pop(random.randint(0, len(self.apollo_pool) - 1))]
            return xs


class AlwenMixedDataset(Dataset):
    def __init__(self,
                 cu_image_path: str,
                 cu_seg_mask_path: str,
                 ap_image_path: str,
                 ap_seg_mask_path: str,
                 preprocessing: callable = None,
                 disable_apollo: bool = False,
                 is_test=False,
                 ih=800,
                 iw=480,
                 ):

        self.culane = CULaneDatasetMK4(cu_image_path, cu_seg_mask_path,
                                       preprocessing=preprocessing,
                                       is_test=is_test)
        self.apollo = ApolloDataset(ih, iw,
                                    arrow_path=ap_image_path,
                                    arrow_mask_path=ap_seg_mask_path,
                                    preprocessing=preprocessing,
                                    is_test=is_test)
        if is_test:
            self.sk = min(len(self.culane), len(self.apollo))
        else:
            self.sk = min(len(self.culane), len(self.apollo))
        if disable_apollo:
            self.total = len(self.culane)
        else:
            self.total = self.sk * 2
        self.divide = self.sk
        self.culane_idx = list(range(len(self.culane)))
        self.apollo_idx = list(range(len(self.apollo)))

        self.culane_pool = random.choices(self.culane_idx, k=len(self.culane))
        self.apollo_pool = random.choices(self.apollo_idx, k=len(self.apollo))

    def __len__(self):
        return self.total

    def __getitem__(self, item):
        if item < self.divide:
            if len(self.culane_pool) == 0:
                self.culane_pool = random.choices(self.culane_idx, k=len(self.culane))
            xs, xt = self.culane[self.culane_pool.pop(random.randint(0, len(self.culane_pool) - 1))]
            return xs, xt
        else:
            if len(self.apollo_pool) == 0:
                self.apollo_pool = random.choices(self.apollo_idx, k=len(self.apollo))
            xs = self.apollo[self.apollo_pool.pop(random.randint(0, len(self.apollo_pool) - 1))]
            return xs
