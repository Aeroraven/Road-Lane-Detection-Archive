import os
import random
import sys
import albumentations as albu
import cv2
import jpeg4py as jpeg
from matplotlib import pyplot as plt
import numpy as np

from torch.utils.data import Dataset
import tqdm

from utils.augmentation import get_tu_simple_augmentation

class ApolloDataset(Dataset):
    def __init__(self,
                 image_size_h: int = 128,
                 image_size_w: int = 128,
                 augmentation: callable = get_tu_simple_augmentation,
                 preprocessing: callable = None,
                 enable_split: bool = True,
                 split_ratio: float = 0.8,
                 is_test: bool = False,
                 arrow_path:str = None,
                 arrow_mask_path: str = None,
                 check_validity: bool = False
                 ):
        if arrow_path is None or arrow_mask_path is None:
            raise Exception()
        self.sr = split_ratio
        self.is_test = is_test
        self.arrow_path = arrow_path
        self.arrow_mask_path = arrow_mask_path
        self.ih = image_size_h
        self.iw = image_size_w
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.seg_image_list = os.listdir(arrow_path)
        print("Total Arrows",len(self.seg_image_list),arrow_path)
        if enable_split:
            random.seed(3407)
            random.shuffle(self.seg_image_list)
            if not is_test:
                self.seg_image_list = self.seg_image_list[:int(len(self.seg_image_list) * self.sr)]
            else:
                self.seg_image_list = self.seg_image_list[int(len(self.seg_image_list) * self.sr):]
        
        
        self.ARROW_COLOR_LIST = [
            [128,  78, 160],  
            [150, 100, 100],  
            [180, 165, 180],  
            [107, 142,  35],  
            [128, 128,   0],  
            [  0,   0, 230]   
        ]
        invalid = 0
        if check_validity:
            print("Checking Validity")
            self.new_seg_list = []
            with tqdm.tqdm(total=len(self),file=sys.stdout,ascii=True) as f:
                for i in range(len(self)):
                    if np.max(self.get_mask_only(i)) == 0:
                        invalid+=1
                    else:
                        self.new_seg_list.append(self.seg_image_list[i])
                    f.set_postfix(invalid=invalid)
                    f.update(1)
                self.seg_image_list = self.new_seg_list
        print("Total samples",len(self))
        print("Invalid Samples:", invalid)


    def albu_resize(self):
        train_transform = albu.Compose([albu.Resize(self.ih, self.iw,interpolation=cv2.INTER_NEAREST)])
        return train_transform

    def __len__(self):
        return len(self.seg_image_list)

    def __getitem__(self, index):
        image = jpeg.JPEG(self.arrow_path + '/' + self.seg_image_list[index]).decode()
        if type(image) is not np.ndarray and not image:
            raise Exception("Invalid file")
        mask = cv2.imread(self.arrow_mask_path + '/' + self.seg_image_list[index].replace(".jpg","_bin.png"))
        # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        if type(mask) is not np.ndarray and not mask:
            raise Exception("Invalid mask",self.arrow_mask_path + '/' + self.seg_image_list[index].replace(".jpg","_bin.png"))
        
        resize_sample = self.albu_resize()(image=image,mask=mask)
        image = resize_sample['image']
        mask = resize_sample['mask']
        mask = self.return_processed_arrows(mask)
        if np.max(mask) == 0:
            return 0
        if self.augmentation and not self.is_test:
            sample = self.augmentation()(image=image)
            image = sample['image']
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image, mask

    def get_mask_only(self,index):
        mask = cv2.imread(self.arrow_mask_path + '/' + self.seg_image_list[index].replace(".jpg","_bin.png"))
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        resize_sample = self.albu_resize()(image=mask)
        mask = resize_sample['image']
        if type(mask) is not np.ndarray and not mask:
            raise Exception("Invalid mask")
        mask = self.return_processed_arrows(mask)
        return mask


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