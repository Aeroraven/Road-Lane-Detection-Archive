import json
import os
import random
from statistics import mode
import sys

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from runner.inference import ProductionModel
from utils.augmentation import get_tu_simple_augmentation
from utils.system import fcpy


class TuSimpleMixedDataset(PyTorchDataset):
    r"""
    Dataset for TuSimple
    """

    def __init__(self,
                 image_path: str,
                 image_size_h: int = 128,
                 image_size_w: int = 128,
                 index_subdirectories = None,
                 augmentation: callable = get_tu_simple_augmentation,
                 preprocessing: callable = None,
                 enable_split: bool = True,
                 split_ratio: float = None,
                 is_test: bool = False,
                 return_original_copy: bool = False,
                 discard_corruption: bool = True,
                 infer_model: str = None,
                 expand_index_range: bool = False,
                 compensate_h: int = 58,
                 compensate_c: int = 5,
                 polyfit_format: bool = False,
                 arrow_path:str = None,
                 arrow_mask_path: str = None
                 ):
        if index_subdirectories is None:
            raise Exception("Subdirectories to be indexed should not be None")
        # Lane Task
        self.indexes = {}
        self.exist_label = {}
        self.top_label = {}
        self.bottom_label = {}
        self.sr = split_ratio
        self.is_test = is_test
        self.return_im_copy = return_original_copy
        self.discard_corruption = discard_corruption
        # Seg Task
        if arrow_path is None or arrow_mask_path is None:
            raise Exception()
        self.arrow_path = arrow_path
        self.arrow_mask_path = arrow_mask_path

        # Lane Task
        if infer_model is None or infer_model == '':
            self.infer_model = None
        else:
            self.infer_model = ProductionModel(infer_model, "onnx", "cpu")
        discarded_samples = 0
        print("Indexing Dataset")
        for i in index_subdirectories:
            label_file = image_path + "/label_data_" + i.split("-")[0] + ".json"
            print("Indexing Dataset", label_file)
            with open(label_file, "r") as f:
                labels = f.read()
            labels_sp = labels.split("\n")
            for j in labels_sp:
                if j == "":
                    continue
                labels_sf = json.loads(j)
                wpx = np.array(labels_sf['lanes']).shape
                lanes = np.array(labels_sf['lanes'])
                
                if lanes.shape[0] <= compensate_c and lanes.shape[1] <= compensate_h:
                    lanes = np.pad(lanes, ((compensate_c - lanes.shape[0], 0), (compensate_h - lanes.shape[1], 0)),
                                   mode='constant', constant_values=-2)
                else:
                    raise Exception("Reading:", labels_sf['raw_file'], " with channels:", lanes.shape[0])

                valid_lanes = 0
                exist_lane = [0 for _ in range(compensate_c)]
                bottom_lane = [0 for _ in range(compensate_c)]
                top_lane = [1e9 for _ in range(compensate_c)]
                for i in range(lanes.shape[0]):
                    for j in range(lanes.shape[1]):
                        if lanes[i, j] != -2:
                            exist_lane[i] = 1
                            bottom_lane[i] = max(bottom_lane[i], j * 10 + 160)
                            top_lane[i] = min(top_lane[i], j * 10 + 160)
                    if exist_lane[i] == 0:
                        top_lane[i] = 0
                        bottom_lane[i] = 0
                mxct = 0
                for i in range(lanes.shape[1]):
                    ct = 0
                    for j in range(lanes.shape[0]):
                        if lanes[j, i] != -2:
                            ct += 1
                    mxct = max(ct, mxct)
                if mxct != sum(exist_lane):
                    if self.discard_corruption:
                        print("Discard ", labels_sf['raw_file'])
                        discarded_samples = discarded_samples + 1
                        continue

                
                if expand_index_range:
                    raw_file_split = labels_sf['raw_file'].split('/')
                    raw_file_prefix = "/".join(raw_file_split[:-1])
                    for k in range(1, 21):
                        self.indexes[image_path + '/' + raw_file_prefix + '/' + str(k) + '.jpg'] = lanes
                        self.exist_label[image_path + '/' + raw_file_prefix + '/' + str(k) + '.jpg'] = np.array(
                            exist_lane)
                        self.top_label[image_path + '/' + raw_file_prefix + '/' + str(k) + '.jpg'] = np.array(top_lane)
                        self.bottom_label[image_path + '/' + raw_file_prefix + '/' + str(k) + '.jpg'] = np.array(
                            bottom_lane)
                else:
                    self.indexes[image_path + '/' + labels_sf['raw_file']] = lanes
                    self.exist_label[image_path + '/' + labels_sf['raw_file']] = np.array(exist_lane)
                    self.top_label[image_path + '/' + labels_sf['raw_file']] = np.array(top_lane)
                    self.bottom_label[image_path + '/' + labels_sf['raw_file']] = np.array(bottom_lane)
        self.image_list = list(self.indexes.keys())
        self.ih = image_size_h
        self.iw = image_size_w
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.polyfit_format = polyfit_format
        print("Available samples", len(self.image_list))
        print("Discarded samples", discarded_samples)
        if enable_split:
            random.seed(3407)
            random.shuffle(self.image_list)
            if not is_test:
                self.image_list = self.image_list[:int(len(self.image_list) * self.sr)]
            else:
                self.image_list = self.image_list[int(len(self.image_list) * self.sr):]
            print("Chosen samples", len(self.image_list))
        print("Processing Dataset")
        self.coef_list = [160 + 10 * i for i in range(compensate_h)]

        # Seg Task
        self.seg_image_list = os.listdir(arrow_path)
        print("Total Arrows",len(self.seg_image_list),arrow_path)
        if enable_split:
            random.seed(3407)
            random.shuffle(self.seg_image_list)
            if not is_test:
                self.seg_image_list = self.seg_image_list[:int(len(self.seg_image_list) * self.sr)]
            else:
                self.seg_image_list = self.seg_image_list[int(len(self.seg_image_list) * self.sr):]
        
        self.divide = len(self.image_list)
        self.total = len(self.seg_image_list) + self.divide
        print("Total samples",self.total)
        self.ARROW_COLOR_LIST = [
            [128,  78, 160],  
            [150, 100, 100],  
            [180, 165, 180],  
            [107, 142,  35],  
            [128, 128,   0],  
            [  0,   0, 230]   
        ]

    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.ih, self.iw)])
        return train_transform

    def __getitem__(self, item):
        if item<self.divide:
            image = jpeg.JPEG(self.image_list[item]).decode()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_org_s = image.shape
            resize_sample = self.albu_resize()(image=image)
            image = resize_sample['image']
            imcopy = image
            if not self.is_test:
                image = self.augmentation()(image=image)['image']
            if self.preprocessing is not None:
                sample = self.preprocessing(image=image)
                image = sample['image']
            coord = np.array(self.indexes[self.image_list[item]])
            if self.infer_model is not None:
                image_mask = self.infer_model.infer(image)[1]
                image_mask[image_mask > 0.5] = 1
                image_mask[image_mask <= 0.5] = 0
                image = image * image_mask
            if self.return_im_copy:
                return image, coord, imcopy, image_org_s
            if self.polyfit_format:
                return image, (coord,
                    self.exist_label[self.image_list[item]].astype("int64"),
                    self.top_label[self.image_list[item]],
                    self.bottom_label[self.image_list[item]])
            return image, coord
        else:
            item = item - self.divide
            image = jpeg.JPEG(self.arrow_path + '\\' + self.seg_image_list[item]).decode()
            if type(image) is not np.ndarray and not image:
                raise Exception("Invalid file")
            mask = cv2.imread(self.arrow_mask_path + '\\' + self.seg_image_list[item].replace(".jpg","_bin.png"))
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            if type(mask) is not np.ndarray and not mask:
                raise Exception("Invalid mask")
            resize_sample = self.albu_resize()(image=image,mask=mask)
            image = resize_sample['image']
            mask = resize_sample['mask']
            mask = self.return_processed_arrows(mask)
            if self.augmentation and not self.is_test:
                sample = self.augmentation()(image=image)
                image = sample['image']
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            return image, mask

    def __len__(self):
        return self.total

    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.ih, self.iw)])
        return train_transform

    def return_processed_arrows(self,image: np.ndarray):
        output = np.zeros((image.shape[0], image.shape[1]))
        for color in self.ARROW_COLOR_LIST:
            image_channel_transposed = np.transpose(image, (2, 0, 1))
            image_thresh_r = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_g = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_b = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_r[image_channel_transposed[2] == color[0]] = 1
            image_thresh_g[image_channel_transposed[1] == color[1]] = 1
            image_thresh_b[image_channel_transposed[0] == color[2]] = 1
            image_thresh_filtered = image_thresh_r * image_thresh_b * image_thresh_g
            output += image_thresh_filtered
        output = np.minimum(output,1)
        return output.astype("int64")