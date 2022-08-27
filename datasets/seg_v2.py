import os
import sys

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from utils.system import fcpy


class SegmentationDatasetVal(PyTorchDataset):
    r"""
    Dataset for semantic segmentation.
    """

    def __init__(self,
                 image_path: str,
                 mask_path: str,
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
                 mask_augmentation: bool = False,
                 tu_simple_trapezoid_filter: bool = False,
                 image_only: bool = False
                 ) -> None:
        """
        Create a common dataset for semantic segmentation
        :param image_size_w: Size of images to be scaled to
        :param image_size_h: Size of images to be scaled to
        :param train_split: Share of runner samples
        :param numpy_dataset: Specify whether the samples are saved in NumPy format
        :param image_path: Path for input images
        :param mask_path: Path for ground truth labels
        :param classes: List of classes. Elements should be integers
        :param augmentation: If exists, apply augmentation transformations to input images
        :param preprocessing: If exists, preprocess input according to the given function
        :param enable_cache: If true, store the preloaded images in the memory. This exhausts
                            resources considerably. Note: augmentations will never be overridden.
        :param enable_file_cache: If true, store the preloaded images in the memory.
                            This allows overriding augmentations
        """
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
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
        self.tu_simple_trapezoid_filter = tu_simple_trapezoid_filter
        self.file_ext = ".png"
        self.image_only = image_only
        if self.tu_simple_trapezoid_filter:
            self.file_ext = ".jpg"
        print("Searching dataset")
        self.image_path_walk = list(self.image_path_walk)
        print("Len",len(self.image_path_walk))
        for i, j, k in tqdm(self.image_path_walk, ascii=True):
            for r in k:
                if self.tu_simple_trapezoid_filter:
                    if r.endswith("20.jpg"):
                        self.image_path_list.append(i[temp + 1:] + "\\" + r)
                else:
                    if r.endswith(".jpg") or r.endswith(".npy") or r.endswith(".PNG"):
                        self.image_path_list.append(i[temp + 1:] + "\\" + r)
        self.image_path_list = self.image_path_list[:int(len(self.image_path_list) * self.train_split)]
        print("Dataset ready. Total files: ", len(self))

    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.image_size_h, self.image_size_w)])
        return train_transform

    def __load_sample__(self, index):
        if self.image_only:
            return self.val_load_impl(index)
        else:
            return self.train_load_impl(index)

    def val_load_impl(self, index):
        org_img = None
        if not self.numpy_dataset:
            if self.enable_file_cache and index in self.image_file_dict:
                image = self.image_file_dict[index]
            else:
                if self.image_path_list[index].endswith(".jpg"):
                    image = jpeg.JPEG(self.image_path + '\\' + self.image_path_list[index]).decode()
                    org_img = image
                else:
                    image = cv2.imread(self.image_path + '\\' + self.image_path_list[index])
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    org_img = image
                if type(image) is not np.ndarray or image is None:
                    raise Exception("Invalid file", self.image_path + '\\' + self.image_path_list[index])
                resize_sample = self.albu_resize()(image=image)
                image = resize_sample['image']
                if self.enable_file_cache:
                    self.image_file_dict[index] = image
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
        else:
            if self.enable_file_cache and index in self.image_file_dict:
                image = self.image_file_dict[index]
            else:
                image = np.load(self.image_path + '\\' + self.image_path_list[index])
                self.image_file_dict[index] = image
        resize_sample = self.albu_resize()(image=org_img)
        org_img = resize_sample['image']
        return org_img, image

    def train_load_impl(self, index):
        if not self.numpy_dataset:
            if self.enable_file_cache and index in self.image_file_dict:
                image = self.image_file_dict[index]
                mask = self.mask_file_dict[index]
            else:
                image = jpeg.JPEG(self.image_path + '\\' + self.image_path_list[index]).decode()
                if type(image) is not np.ndarray and not image:
                    raise Exception("Invalid file", self.image_path + '\\' + self.image_path_list[index])
                mask = cv2.imread(self.mask_path + '\\' + self.image_path_list[index][:-4] + self.file_ext)
                if type(mask) is not np.ndarray and not mask:
                    raise Exception("Invalid mask",
                                    self.mask_path + '\\' + self.image_path_list[index][:-4] + self.file_ext)
                mask = np.transpose(mask, (2, 0, 1))[0]
                mask = np.minimum(mask, 1)
                resize_sample = self.albu_resize()(image=image, mask=mask)
                image, mask = resize_sample['image'], resize_sample['mask']
                if self.enable_file_cache:
                    self.image_file_dict[index] = image
                    self.mask_file_dict[index] = mask
            if self.expanding_onehot:
                masks = [(mask == v) for v in self.classes]
                mask = np.stack(masks, axis=-1).astype("float")
            if self.augmentation:
                if self.mask_augmentation:
                    sample = self.augmentation(image=image, mask=mask)
                    image = sample['image']
                    mask = sample['mask']
                else:
                    sample = self.augmentation(image=image)
                    image = sample['image']
            if self.preprocessing:
                if self.mask_augmentation:
                    sample = self.preprocessing(image=image, mask=mask)
                    image = sample['image']
                    mask = sample['mask']
                else:
                    sample = self.preprocessing(image=image)
                    image = sample['image']
                    mask = np.transpose(mask, (1, 0)).astype("int64")
        else:
            if self.enable_file_cache and index in self.image_file_dict:
                image = self.image_file_dict[index]
                mask = self.mask_file_dict[index]
            else:
                image = np.load(self.image_path + '\\' + self.image_path_list[index])
                mask = np.load(self.mask_path + '\\' + self.image_path_list[index])
                self.image_file_dict[index] = image
                self.mask_file_dict[index] = mask

        return image, mask

    def cache_dataset(self, index):
        """
        Cache the dataset
        :param index: Index of the sample
        :return: Image and the corresponding mask for runner
        """
        if self.enable_cache and index not in self.image_dict:
            self.image_dict[index], self.mask_dict[index] = self.__load_sample__(index)
        return self.image_dict[index], self.mask_dict[index]

    def __getitem__(self, item):
        if self.enable_cache:
            return self.cache_dataset(item)
        else:
            return self.__load_sample__(item)

    def __len__(self):
        return len(self.image_path_list)

    def save_to_np_file(self, image_path, mask_path):
        """
        Save the dataset in NumPy format
        :param image_path: Output folder for images
        :param mask_path:  Output folder for ground truths
        """
        length = len(self)
        for i in tqdm(range(length), file=sys.stdout, desc="Converting Image Format"):
            image, mask = self[i]
            np.save(image_path + "/" + self.image_path_list[i][:-4].replace("/", "_").replace("\\", "_") + ".npy",
                    image)
            np.save(mask_path + "/" + self.image_path_list[i][:-4].replace("/", "_").replace("\\", "_") + ".npy", mask)

    def save_to_img_file(self, image_path, mask_path, mask_ext=".png"):
        for i in tqdm(range(len(self)), file=sys.stdout, desc="Processing Dataset", ascii=True):
            fcpy(self.image_path + '\\' + self.image_path_list[i],
                 image_path + "/" + self.image_path_list[i][:-4].replace("/", "_").replace("\\", "_") + ".jpg",
                 remove=False)
            fcpy(self.mask_path + '\\' + self.image_path_list[i][:-4] + mask_ext,
                 mask_path + "/" + self.image_path_list[i][:-4].replace("/", "_").replace("\\", "_") + mask_ext,
                 remove=False)
