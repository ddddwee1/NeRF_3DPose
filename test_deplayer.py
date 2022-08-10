import os
import cv2
import yaml
import torch
import numpy as np 

from tqdm import tqdm
from easydict import EasyDict
from lib import dataloader, trainer

def convert_batch_to_torch(batch):
    res = {}
    for k in batch:
        a = np.array(batch[k])
        if len(a.shape)==0:
            a = a.reshape(1)
        res[k] = torch.from_numpy(a)
    return res

def render_img(weights, z_vals, idx):
    idx = idx.cpu().numpy()
    z_vals = z_vals.cpu().numpy()
    zmin = z_vals.min()
    zmax = z_vals.max()
    weights = weights.cpu().numpy()
    
    # get the argmax depth
    canvas = np.zeros([z_vals.shape[-1], cfg.DATA.img_size * cfg.DATA.img_size], dtype=np.uint8)
    for i in range(len(idx)):
        ii = idx[i]
        w = weights[i]
        # print(w.max())
        for a in range(w.shape[0]):
            wval = w[a]
            pixel_val = np.uint8(np.clip(wval * 255, 0, 255))
            canvas[a, ii] = pixel_val
    canvas = canvas.reshape([-1, cfg.DATA.img_size, cfg.DATA.img_size])
    return canvas

cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))
dataset = dataloader.SPMLDataset(cfg, is_training=False)
pose_metadata = dataset.get_meta()

nerf_trainer = trainer.Trainer(cfg, pose_metadata)
nerf_trainer.initialize()

os.makedirs(f'{cfg.EXPERIMENT.name}/depth_layer/', exist_ok=True)
# for i in tqdm(range(len(dataset))):
#     data = dataset[i]
#     data = convert_batch_to_torch(data)
#     with torch.no_grad():
#         weights, z_vals = nerf_trainer.evaluate_depth(data)
#     result = render_img(weights, z_vals, data['sampled_idx'])
#     # print(result.shape)
#     cv2.imwrite(f'{cfg.EXPERIMENT.name}/depth/{i:06d}.png', result)

i = 0
data = dataset[i]
data = convert_batch_to_torch(data)
with torch.no_grad():
    weights, pts = nerf_trainer.evaluate_depth(data)
    z_vals = pts[..., 2]
    print(z_vals.shape)
result = render_img(weights, z_vals, data['sampled_idx'])

for i in range(result.shape[0]):
    cv2.imwrite(f'{cfg.EXPERIMENT.name}/depth_layer/{i:06d}.png', result[i])

