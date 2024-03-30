import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
import trimesh
import random
from collections import defaultdict
import fd.config
import fn.config
import fn.checkpoints
import fd.checkpoints
from generation import Generator3D6
import numpy as np


def farthest_point_sample(xyz, pointnumber):
    device = 'cuda'
    N, C = xyz.shape
    torch.seed()
    xyz = torch.from_numpy(xyz).float().to(device)
    centroids = torch.zeros(pointnumber, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    farthest[0] = N/2
    for i in tqdm(range(pointnumber)):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.detach().cpu().numpy().astype(int)


tpointnumber = [4096, 8192, 16384]

datalist = ['withoutNormals/2048/cylinder100k_2048.xyz',
            'withoutNormals/2048/galera100k_2048.xyz',
            'withoutNormals/2048/netsuke100k_2048.xyz',
            ]

outlist1 = ['PCP_results/4096/cylinder100k_2048.xyz',
            'PCP_results/4096/galera100k_2048.xyz',
            'PCP_results/4096/netsuke100k_2048.xyz',
            ]
outlist2 = ['PCP_results/8192/cylinder100k_2048.xyz',
            'PCP_results/8192/galera100k_2048.xyz',
            'PCP_results/8192/netsuke100k_2048.xyz',
            ]
outlist3 = ['PCP_results/16384/cylinder100k_2048.xyz',
            'PCP_results/16384/galera100k_2048.xyz',
            'PCP_results/16384/netsuke100k_2048.xyz',
            ]

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = 'out/pointcloud/opu'

cfg1 = fn.config.load_config('configs/fn.yaml')
cfg2 = fd.config.load_config('configs/fd.yaml')

model = fn.config.get_model(cfg1, device)
model2 = fd.config.get_model(cfg2, device)

checkpoint_io1 = fn.checkpoints.CheckpointIO('out/fn', model=model)
load_dict = checkpoint_io1.load('model_best.pt')

checkpoint_io2 = fd.checkpoints.CheckpointIO('out/fd', model=model2)
load_dict = checkpoint_io2.load('model_best.pt')

model.eval()
model2.eval()

generator = Generator3D6(model, model2, device)


for k in range(len(datalist)):
    print("processing "+datalist[k])
    # normalization
    xyzname = datalist[k]
    cloud = np.loadtxt(xyzname)
    cloud = cloud[:, 0:3]
    bbox = np.zeros((2, 3))
    bbox[0][0] = np.min(cloud[:, 0])
    bbox[0][1] = np.min(cloud[:, 1])
    bbox[0][2] = np.min(cloud[:, 2])
    bbox[1][0] = np.max(cloud[:, 0])
    bbox[1][1] = np.max(cloud[:, 1])
    bbox[1][2] = np.max(cloud[:, 2])
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale1 = 1/scale
    for i in range(cloud.shape[0]):
        cloud[i] = cloud[i]-loc
        cloud[i] = cloud[i]*scale1
    np.savetxt("test.xyz", cloud)
    cloud = np.expand_dims(cloud, 0)

    # upsampling
    pointcloud = np.array(generator.upsample(cloud))

    # farthest_point_sample
    print("farthest point sample")
    for i in range(pointcloud.shape[0]):
        pointcloud[i] = pointcloud[i]*scale
        pointcloud[i] = pointcloud[i]+loc

    centroids1 = farthest_point_sample(pointcloud, tpointnumber[0])

    if not os.path.exists(f"PCP_results/4096/"):
        os.makedirs(f"PCP_results/4096/")

    np.savetxt(outlist1[k], pointcloud[centroids1])
    np.savetxt(outlist2[k], pointcloud[centroids2])
    np.savetxt(outlist3[k], pointcloud[centroids3])
    print("done")
