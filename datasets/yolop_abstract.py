import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from jpeg4py import JPEG
from PIL import Image
from torch.utils.data import Dataset
from utils.augmentation import letterbox, augment_hsv, random_perspective


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """

    def __init__(self,
                 image_path,
                 mask_path,
                 is_train,
                 inputsize=640,
                 transform=None):
        """
        initial all the characteristic
        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(image_path)
        lane_root = Path(mask_path)
        self.img_root = img_root
        self.lane_root = lane_root
        self.mask_list = list(self.lane_root.iterdir())

        self.db = []

        self.data_format = ".jpg"
        self.scale_factor = 0.25
        self.rotation_factor = 10
        self.flip = True
        self.color_rgb = False

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array([720,1280])

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self, ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx]
        img = JPEG(data['image']).decode()
        lane_label = cv2.imread(data["lane"], 0)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)

        (img, lane_label), ratio, pad = letterbox((img, lane_label), resized_shape, auto=True,
                                                             scaleup=self.is_train)
        if self.is_train:
            combination = (img, lane_label)
            (img, lane_label) = random_perspective(
                combination=combination,
                targets=(),
                degrees=10,
                translate=0.1,
                scale=0.25,
                shear=0
            )
            augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)


            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                lane_label = np.fliplr(lane_label)

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                lane_label = np.filpud(lane_label)

        img = np.ascontiguousarray(img)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
        lane_label = torch.stack((lane2[0], lane1[0]), 0)
        target = lane_label
        img = self.transform(img)
        return img, target

    def select_data(self, db):
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0),
                                     torch.stack(label_lane, 0)], paths,