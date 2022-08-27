import torch.utils.data as torch_data
import numpy as np


class AlwenMockDataset(torch_data.Dataset):
    def __init__(self):
        super(AlwenMockDataset, self).__init__()
        self.total = 4800
        self.divide = 2400

    def __getitem__(self, item):
        if item < self.divide:
            image = np.random.random((3,480,800)).astype("float32")
            label = np.random.random((480,800)).astype("int64")
            return image,label
        else:
            image = np.random.random((3,480,800)).astype("float32")
            label = np.random.randint(1,1200,(5,58)).astype("int64")
            return image, label

    def __len__(self):
        return self.total
