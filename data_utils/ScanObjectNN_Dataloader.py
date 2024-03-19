import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from PillarCoder.model.Tool import pc_normalize


class ScanObjectNN(Dataset):
    def __init__(self, path):
        super(ScanObjectNN, self).__init__()
        file = h5py.File(path, 'r')
        self.data = np.array(file["data"])
        self.labels = np.array(file["label"])

        # label weights
        self.num_labels = np.bincount(self.labels.reshape((-1))).astype(np.float32)
        self.label_weights = torch.from_numpy(np.power(np.amax(self.num_labels) / self.num_labels, 1 / 3.0))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return pc_normalize(self.data[item]), self.labels[item]
