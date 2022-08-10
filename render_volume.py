import os
import cv2
import yaml
import torch
import pickle
import numpy as np 

from tqdm import tqdm
from easydict import EasyDict
from lib import dataloader, trainer
from scipy.interpolate import griddata

def convert_batch_to_torch(batch):
    res = {}
    for k in batch:
        a = np.array(batch[k])
        if len(a.shape)==0:
            a = a.reshape(1)
        res[k] = torch.from_numpy(a)
    return res

def fit_grid(weights, pts):
    pts = pts.cpu().numpy()
    pts = pts.reshape([-1,3])
    pts = pts[::4]
    weights = weights.cpu().numpy()
    weights = weights.reshape(-1)
    weights = weights[::4]
    xmin = pts[...,0].min()
    xmax = pts[...,0].max()
    ymin = pts[...,1].min()
    ymax = pts[...,1].max()
    zmin = pts[...,2].min()
    zmax = pts[...,2].max()
    grid_x, grid_y, grid_z = np.mgrid[xmin:xmax:40j, ymin:ymax:40j, zmin:zmax:40j]
    print('FITTING...')
    grid_values = griddata(pts, weights, (grid_x, grid_y, grid_z), method='linear', fill_value=0)
    return grid_values, [grid_x, grid_y, grid_z]

cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))
dataset = dataloader.SPMLDataset(cfg, is_training=False)
pose_metadata = dataset.get_meta()

nerf_trainer = trainer.Trainer(cfg, pose_metadata)
nerf_trainer.initialize()

os.makedirs(f'{cfg.EXPERIMENT.name}/grid_data/', exist_ok=True)
i = 0
data = dataset[i]
data = convert_batch_to_torch(data)
with torch.no_grad():
    weights, pts = nerf_trainer.evaluate_depth(data)
result, grids = fit_grid(weights, pts)
pickle.dump([result, grids], open(f'{cfg.EXPERIMENT.name}/grid_data/{i:06d}.pkl', 'wb'))
