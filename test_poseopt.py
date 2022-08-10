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

def render_img(rgb, idx):
    canvas = np.zeros([cfg.DATA.img_size * cfg.DATA.img_size, 3], dtype=np.uint8)
    idx = idx.cpu().numpy()
    rgb = rgb.cpu().numpy()
    rgb = np.uint8(rgb * 255)
    for i in range(len(idx)):
        ii = idx[i]
        canvas[ii] = rgb[i]
    canvas = canvas.reshape([cfg.DATA.img_size, cfg.DATA.img_size, 3])
    return canvas

cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))
dataset = dataloader.SPMLDataset(cfg, is_training=False)
pose_metadata = dataset.get_meta()

nerf_trainer = trainer.Trainer(cfg, pose_metadata)
nerf_trainer.initialize()

os.makedirs(f'{cfg.EXPERIMENT.name}/visualization/', exist_ok=True)
for i in tqdm(range(len(dataset))):
    data = dataset[i]
    data = convert_batch_to_torch(data)
    with torch.no_grad():
        rgb = nerf_trainer.evaluate(data)
    result = render_img(rgb, data['sampled_idx'])
    # print(result.shape)
    cv2.imwrite(f'{cfg.EXPERIMENT.name}/visualization/{i:06d}.png', result)
