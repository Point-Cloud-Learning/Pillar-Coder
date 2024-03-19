import os
import json
import numpy as np

from torch.utils.data import Dataset
from PillarCoder.model.Tool import pc_normalize, farthest_point_sample


class ShapeNet(Dataset):
    def __init__(self, root="data/shapenetcore_partanno_segmentation_benchmark_v0_normal/", num_point=1024, split="train_val", class_choice=None, use_normals=False,
                 uniform_sampling="farthest_point_sample"):
        super(ShapeNet, self).__init__()
        self.root = root
        self.num_point = num_point
        self.cat_file = os.path.join(self.root, 'synsetoffset2category.txt')
        self.use_normals = use_normals
        self.uniform_sampling = uniform_sampling

        self.cat = {}
        with open(self.cat_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        if class_choice:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'train_val':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.data_path = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.data_path.append((item, fn))

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        if item in self.cache:
            point_set, seg, cls = self.cache[item]
        else:
            fn = self.data_path[item]
            cat = fn[0]
            cls = self.classes[cat]
            cls = np.array(cls).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if self.use_normals:
                point_set = data[:, 0:6]
            else:
                point_set = data[:, 0:3]
            seg = data[:, -1].astype(np.int32)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if self.uniform_sampling == "random":
                self.cache[item] = (point_set, seg, cls)

            elif self.uniform_sampling == "farthest_point_sample":
                centroids = farthest_point_sample(point_set, self.num_point)
                point_set = point_set[centroids, :]
                seg = seg[centroids]
                self.cache[item] = (point_set, seg, cls)

        if self.uniform_sampling == "random":
            # replace=False means non-repetitive selection
            choice = np.random.choice(len(seg), self.num_point)
            point_set = point_set[choice, :]
            seg = seg[choice]

        return point_set, seg, cls
