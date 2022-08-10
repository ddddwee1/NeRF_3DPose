import torch
import numpy as np 

def get_rays_from_camera(img_size, focal=10000):
    # we set the camera to 0,0
    rays_o = np.zeros([img_size, img_size, 3], dtype=np.float32)
    i,j = np.meshgrid(np.arange(img_size, dtype=np.float32),
                        np.arange(img_size, dtype=np.float32), indexing='xy')
    rays_d = np.stack([i/focal, j/focal, np.ones_like(i)], -1)
    return rays_o, rays_d

