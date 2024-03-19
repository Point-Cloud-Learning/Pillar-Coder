import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PillarCoder.model.Tool import farthest_point_sample, pc_normalize


class ModelNet40(Dataset):
    def __init__(self, root="./data/modelnet40_normal_resampled/", num_point=1024, split="train", uniform_sampling="random", use_normals=False):
        super(ModelNet40, self).__init__()
        self.root = root
        self.num_point = num_point
        self.use_normals = use_normals
        self.uniform_sampling = uniform_sampling

        self.cat_file = os.path.join(self.root, "modelnet40_shape_names.txt")
        self.cat = [line.rstrip() for line in open(self.cat_file)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {"train": [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_train.txt"))],
                     "test": [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_test.txt"))]}

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]

        self.cache = {}

        # label weights
        self.num_labels = np.zeros(len(self.classes))
        for cls in self.data_path:
            self.num_labels[self.classes[cls[0]]] += 1
        self.num_labels = self.num_labels.astype(np.float32)
        label_weights = self.num_labels / np.sum(self.num_labels)
        self.label_weights = torch.from_numpy(np.power(np.amax(label_weights) / label_weights, 1 / 3.0))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        if item in self.cache:
            point_set, cls = self.cache[item]
        else:
            fn = self.data_path[item]
            cls = self.classes[fn[0]]
            cls = np.array(cls).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if self.uniform_sampling == "farthest_point_sample":
                point_set = farthest_point_sample(point_set, self.num_point)
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.cache[item] = (point_set, cls)

            elif self.uniform_sampling == "random&saving":
                choice = np.random.choice(len(point_set), self.num_point, replace=False)
                point_set = point_set[choice, :]
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.cache[item] = (point_set, cls)

            elif self.uniform_sampling == "random":
                self.cache[item] = (point_set, cls)

            else:
                point_set = point_set[0:self.num_point, :]
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.cache[item] = (point_set, cls)

        if self.uniform_sampling == "random":
            choice = np.random.choice(len(point_set), self.num_point, replace=False)
            point_set = point_set[choice, :]
            if not self.use_normals:
                point_set = point_set[:, 0:3]

        return point_set, cls
