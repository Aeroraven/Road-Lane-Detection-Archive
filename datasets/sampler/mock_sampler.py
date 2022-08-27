import random

import torch.utils.data as torch_data

from datasets.mock_dataset import AlwenMockDataset


class AlwenMockSampler(torch_data.Sampler):
    def __init__(self, data: AlwenMockDataset):
        super(AlwenMockSampler, self).__init__(data)
        self.data = data
        self.index_1 = list(range(0, data.divide))
        self.index_2 = list(range(data.divide, data.total))

    def __iter__(self):
        random.shuffle(self.index_1)
        random.shuffle(self.index_2)
        return iter(self.index_1)

    def __len__(self):
        return len(self.index_1) + len(self.index_2)


class AlwenMockBatchSampler(torch_data.Sampler):
    def __init__(self, sampler:AlwenMockSampler, batch_size, drop_last):
        super(AlwenMockBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        batch = []
        for idx in self.sampler.index_1:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if len(batch)!=0:
            batches.append(batch)
            batch = []
        for idx in self.sampler.index_2:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if len(batch)!=0:
            batches.append(batch)
            batch = []
        random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size