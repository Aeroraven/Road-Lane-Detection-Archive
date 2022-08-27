from torch.utils.data.dataset import Dataset
import os
import json
import numpy as np
import cv2
from  matplotlib import pyplot as plt
# Contributor: HugePotatoMonster

# 修改此处以匹配本机的数据集位置
DATASET_PATH = {
    'tusimple':"D:\PM\Dataset\TuSimple",
    'apollo':"D:\PM\Dataset\Apollo"
}

TUSIMPLE_SUBFOLDER = {
    'train': 'train_set',
    'test': 'test_set'
}

TUSIMPLE_SPLIT_FILES = {
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'test': ['test_tasks_0627.json'],
}

APOLLO_SUBFOLDER = {
    'data': 'ColorImage_road02\ColorImage',
    'label': 'Labels_road02\Label',
    'camera': 'Camera 5'
}

# 修改此处以调整训练/测试集的大小
APOLLO_DATASET_SEG = {
    'train_record_num' : 30,
    'train_pic_num' : 100,
    'test_record_num' : 4,
    'test_pic_num' : 100
}

# 箭头RGB
ARROW_COLOR_LIST = [
    [128,  78, 160],  # 前行/左转
    [150, 100, 100],  # 前行/右转
    [180, 165, 180],  # 左转
    [107, 142,  35],  # 右转
    [128, 128,   0],  # 前行
    [  0,   0, 230]   # 掉头
]

class CombinedDataset(Dataset):

    # split: 'train' or 'test'
    def __init__(self, split='train'):
        # 信息列表
        self.tusimple_list = []
        self.apollo_list = []
        # train/test
        self.split = split
        # 记录tusimple最后一个元素的位置
        self.divide = -1
        # 加载数据
        self.load_tusimple_list()
        self.load_apollo_list()
        self.total = len(self)
        self.divide = self.divide+1

    # tusimple信息加载
    def load_tusimple_list(self):
        print("Tusimple Preparing...")

        for file_name in TUSIMPLE_SPLIT_FILES[self.split]:
            file_path = os.path.join(DATASET_PATH['tusimple'],TUSIMPLE_SUBFOLDER[self.split],file_name)
            with open(file_path, 'r') as f:
                while True:
                    line = f.readline()
                    if line=='':
                        f.close()
                        break
                    label = json.loads(line)
                    pic_path = os.path.join(DATASET_PATH['tusimple'],TUSIMPLE_SUBFOLDER[self.split],label['raw_file'])
                    lanes = np.array(label["lanes"])
                    # 图片路径&车道信息
                    self.tusimple_list.append([pic_path,lanes])
                    self.divide += 1

        print("Finished.",len(self.tusimple_list)," record loaded")

    # apollo信息加载
    def load_apollo_list(self):
        print("Apollo Preparing...")

        record_num = self.split+'_record_num'
        pic_num = self.split+'_pic_num'

        data_dir = os.path.join(DATASET_PATH['apollo'],APOLLO_SUBFOLDER['data'])
        record_dirs = os.listdir(data_dir)

        # 测试集从后往前取
        if self.split=='test':
            record_dirs.reverse()

        for i in range(APOLLO_DATASET_SEG[record_num]):
            record_dir = record_dirs[i]
            pic_dir = os.path.join(data_dir, record_dir, APOLLO_SUBFOLDER['camera'])
            pic_name_list = os.listdir(pic_dir)
            num = min(APOLLO_DATASET_SEG[pic_num],len(pic_name_list))
            for j in range(num):
                pic_name = pic_name_list[j]
                pic_path = os.path.join(pic_dir,pic_name)
                label_path = os.path.join(DATASET_PATH['apollo'],APOLLO_SUBFOLDER['label'],record_dir,APOLLO_SUBFOLDER['camera'],pic_name)[:-4]+'_bin.png'
                self.apollo_list.append([pic_path,label_path])
        print("Finished.",len(self.apollo_list)," record loaded")

    def __getitem__(self, index):
        data_type = 0
        # Tusimple
        if index<=self.divide:
            img = cv2.imread(self.tusimple_list[index][0])
            label = self.tusimple_list[index][1]
            data_type = 1
        # Apollo
        else:
            # img: W*H*Channels
            # label: W*H
            # data_type: 2
            img = cv2.imread(self.apollo_list[index-self.divide-1][0])
            label_img = cv2.imread(self.apollo_list[index-self.divide-1][1])
            label = self.return_processed_arrows(label_img)
            data_type = 2
        return img, label

    def return_processed_arrows(self,image: np.ndarray):
        output = np.zeros((image.shape[0], image.shape[1]))
        for color in ARROW_COLOR_LIST:
            image_channel_transposed = image
            image_thresh_r = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_g = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_b = np.zeros((image.shape[0], image.shape[1]))
            image_thresh_r[image_channel_transposed[:,:,2] == color[0]] = 1
            image_thresh_g[image_channel_transposed[:,:,1] == color[1]] = 1
            image_thresh_b[image_channel_transposed[:,:,0] == color[2]] = 1
            image_thresh_filtered = image_thresh_r * image_thresh_b * image_thresh_g
            output += image_thresh_filtered
        output = np.minimum(output,1)
        return output

    def __len__(self):
        return len(self.tusimple_list)+len(self.apollo_list)