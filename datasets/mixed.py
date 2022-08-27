from torch.utils.data import Dataset
import jpeg4py as jpeg
import cv2
import numpy as np
import albumentations as albu
import os

from utils.system import file_readlines


class MixedDataset(Dataset):
    def __init__(self,
                 image_path:str,
                 mask_path:str,
                 line_path:str,
                 classes: list,
                 augmentation: callable = None,
                 preprocessing: callable = None,
                 enable_cache: bool = False,
                 enable_file_cache: bool = False,
                 image_size_h: int = 128,
                 image_size_w: int = 128,
                 train_split: float = 1.0,
                 numpy_dataset: bool = True,
                 expanding_onehot: bool = False,
                 mask_augmentation: bool = False
                 ):
        super(MixedDataset, self).__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.line_path = line_path
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.enable_cache = enable_cache
        self.enable_file_cache = enable_file_cache
        self.image_dict = {}
        self.mask_dict = {}
        self.image_file_dict = {}
        self.mask_file_dict = {}
        self.image_path_list = []
        self.image_path_walk = os.walk(self.image_path)
        temp = len(self.image_path)
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.train_split = train_split
        self.numpy_dataset = numpy_dataset
        self.expanding_onehot = expanding_onehot
        self.mask_augmentation = mask_augmentation
        print("Searching dataset")
        for i, j, k in self.image_path_walk:
            for r in k:
                if r.endswith(".jpg") or r.endswith(".npy"):
                    self.image_path_list.append(i[temp + 1:] + "\\" + r)
        self.image_path_list = self.image_path_list[:int(len(self.image_path_list) * self.train_split)]
        print("Dataset ready. Total files: ", len(self))

    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.image_size_h, self.image_size_w)])
        return train_transform

    def __getitem__(self, index):
        image = jpeg.JPEG(self.image_path + '\\' + self.image_path_list[index]).decode()
        if type(image) is not np.ndarray and not image:
            raise Exception("Invalid file", self.image_path + '\\' + self.image_path_list[index])
        mask = cv2.imread(self.mask_path + '\\' + self.image_path_list[index][:-4] + ".png")
        if type(mask) is not np.ndarray and not mask:
            raise Exception("Invalid mask", self.mask_path + '\\' + self.image_path_list[index][:-4] + ".png")
        mask = np.transpose(mask, (2, 0, 1))[0]
        resize_sample = self.albu_resize()(image=image, mask=mask)
        image, mask = resize_sample['image'], resize_sample['mask']
        mask = np.squeeze(mask)
        line = file_readlines(self.mask_path + '\\' + self.image_path_list[index][:-4] + ".txt")
        line_ex = np.array([(i<=len(line)) for i in range(len(self.classes))]).astype("int64")
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image, mask, line_ex

    def __len__(self):
        return len(self.image_path_list)