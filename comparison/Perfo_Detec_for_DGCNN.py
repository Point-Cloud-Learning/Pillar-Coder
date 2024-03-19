import argparse

import torch
import logging
import numpy as np

from torch import nn
from tqdm import tqdm
from thop import profile, clever_format
from torch.backends import cudnn
from PillarCoder.Train_Scan import generate_groups
from PillarCoder.model.Tool import square_distance
from model.Unifier import get_unifier_cls, get_unifier_par, get_unifier_sem
from data_utils.ModelNet40_Dataloader import ModelNet40
from data_utils.ScanObjectNN_Dataloader import ScanObjectNN
from data_utils.ShapeNet_Dataloader import ShapeNet
from data_utils.S3DIS_Dataloader import S3DIS

from model_DGCNN import DGCNN_cls, DGCNN_partseg, DGCNN_semseg_s3dis


parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['dgcnn'],
                    help='Model to use, [dgcnn]')
parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                    choices=['S3DIS'])
parser.add_argument('--test_area', type=str, default=None, metavar='N',
                    choices=['1', '2', '3', '4', '5', '6', 'all'])
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=4096,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_root', type=str, default='', metavar='N',
                    help='Pretrained model root')
parser.add_argument('--visu', type=str, default='',
                    help='visualize the model')
parser.add_argument('--visu_format', type=str, default='ply',
                    help='file format of visualization')
args = parser.parse_args()


def get_logger(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=log_dir + "Perfo_Detec.txt", mode='a', encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt=fmt), sh.setFormatter(fmt=fmt)
    log.addHandler(fh), log.addHandler(sh)
    return log


def cal_flops_params(model, inputs):
    flops, params = profile(model=model, inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


def cal_latency(model, inputs):
    repetitions = 300
    cudnn.benchmark = True
    # warmup
    print('warm up ...')
    with torch.no_grad():
        for _ in range(100):
            _ = model(*inputs)
    # statistics
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    print('testing ...')
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(*inputs)
            ender.record()
            # waiting GPU
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    avg = timings.sum() / repetitions
    return avg


def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y, ]


"""close all batch normalization layers in the model when testing due to only one sample is fed into the model, else raising an error"""
if __name__ == "__main__":
    # cls_ModelNet, cls_ScanObjectNN, par_ShapeNet, sem_S3DIS
    obj = "sem_S3DIS"
    logger = get_logger("./logs/")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if obj == "cls_ModelNet":
        """get model"""
        model = DGCNN_cls(args)
        model = model.to(device)
        model = model.eval()

        """prepare inputs"""
        modelnet = ModelNet40(uniform_sampling="random", use_normals=False)
        features, _ = modelnet[0]
        features = torch.from_numpy(features)[None, ].to(device)
        features = features.transpose(2, 1)
        inputs = (features, )

        """calculate flops and params"""
        flops, params = cal_flops_params(model, inputs)

        """calculate latency"""
        avg_latency = cal_latency(model, inputs)

        logger.info("DGCNN - %s - flops: %s, params: %s, latency: %.4fms \n" % (obj, flops, params, avg_latency))

    elif obj == "par_ShapeNet":
        """get model"""
        model = DGCNN_partseg(args, 50)
        model = model.to(device)
        model.eval()

        """prepare inputs"""
        shapenet = ShapeNet()
        features, _, cls = shapenet[0]
        features = torch.from_numpy(features)[None, ].to(device)
        features = features.transpose(2, 1)
        classes = to_categorical(cls[None, ], 16).to(device)
        inputs = (features, classes)

        """calculate flops and params"""
        flops, params = cal_flops_params(model, inputs)

        """calculate latency"""
        avg_latency = cal_latency(model, inputs)

        logger.info("DGCNN - %s - flops: %s, params: %s, latency: %.4fms \n" % (obj, flops, params, avg_latency))

    else:
        """get model"""
        model = DGCNN_semseg_s3dis(args)
        model = model.to(device)
        model.eval()

        """prepare inputs"""
        s3dis = S3DIS()
        features, _ = s3dis[0]
        features = torch.from_numpy(features)[None, ].to(device)
        features = features.transpose(2, 1)
        inputs = (features, )

        """calculate flops and params"""
        flops, params = cal_flops_params(model, inputs)

        """calculate latency"""
        avg_latency = cal_latency(model, inputs)

        logger.info("DGCNN - %s - flops: %s, params: %s, latency: %.4fms \n" % (obj, flops, params, avg_latency))
