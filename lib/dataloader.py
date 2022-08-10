import os
import cv2
import torch
import pickle
import random
import numpy as np 

from lib.utils import camera
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset

class SPMLDataset(object):
	def __init__(self, cfg, is_training):
		super(SPMLDataset, self).__init__()
		self.cfg = cfg
		self.is_training = is_training
		
		self.idx_sel = np.load(os.path.join(cfg.DATA.dataset_path, 'selected_idx.npy'))
		self.meta = pickle.load(open(os.path.join(cfg.DATA.dataset_path, 'meta.pkl'), 'rb'))
		self.rays_o, self.rays_d = camera.get_rays_from_camera(cfg.DATA.img_size, cfg.DATA.focal)
		# print(self.rays_o.shape, self.rays_d.shape)
		self.rays_o = self.rays_o.reshape([-1,3])
		self.rays_d = self.rays_d.reshape([-1,3])

	def __len__(self):
		return len(self.idx_sel)

	def __getitem__(self, idx0):
		idx = self.idx_sel[idx0]
		meta = self.meta[idx]
		pose = meta['pose']
		rest_joints = meta['rest_joints']
		translation = meta['translation']
		near = meta['near'].astype(np.float32)
		far = meta['far'].astype(np.float32)

		img = cv2.imread(os.path.join(self.cfg.DATA.dataset_path, 'images', '%06d.png'%idx))
		mask = cv2.imread(os.path.join(self.cfg.DATA.dataset_path, 'masks', '%06d.png'%idx), cv2.IMREAD_GRAYSCALE)
		rgb, rays_o, rays_d, sampled_idx = self._select_pixel(img, mask)
		if self.is_training:
			return {'RGB':rgb, 'rays_o':rays_o, 'rays_d':rays_d, 'near':near, 'far':far, 'idx':idx0}
		else:
			return {'RGB':rgb, 'rays_o':rays_o, 'rays_d':rays_d, 'near':near, 'far':far, 'idx':idx0, 'sampled_idx': sampled_idx}

	def _select_pixel(self, img, mask):
		mask = mask.reshape([self.cfg.DATA.img_size * self.cfg.DATA.img_size])
		valid_idx, = np.where(mask==255)
		if self.is_training:
			sampled_idx = np.random.choice(valid_idx, self.cfg.DATA.n_rays_per_image, replace=False)
			sampled_idx = np.sort(sampled_idx)
		else:
			sampled_idx = valid_idx

		img = img.reshape([-1, 3])
		rgb = img[sampled_idx]
		rays_o = self.rays_o[sampled_idx]
		rays_d = self.rays_d[sampled_idx]
		rgb = np.float32(rgb) / 255
		return rgb, rays_o, rays_d, sampled_idx

	def get_meta(self):
		all_poses = []
		all_rest_joints = []
		all_translation = []
		for idx in self.idx_sel:
			all_poses.append(self.meta[idx]['pose'])
			all_rest_joints.append(self.meta[idx]['rest_joints'])
			all_translation.append(self.meta[idx]['translation'])
		all_poses = np.stack(all_poses, axis=0)
		all_rest_joints = np.stack(all_rest_joints, axis=0)
		all_translation = np.stack(all_translation, axis=0).reshape([-1, 1, 3])
		print('Pose metadata:  Poses:', all_poses.shape, 'Rest:', all_rest_joints.shape, 'Translation:', all_translation.shape)
		return {'poses':all_poses.astype(np.float32), 'rest_joints':all_rest_joints.astype(np.float32), 'translation':all_translation.astype(np.float32)}

class RandomSampler(Sampler):
    def __init__(self, bsize, data_len, n_iters):
        self.bsize = bsize
        self.data_len = data_len
        self.n_iters = n_iters
        
    def __iter__(self):
        for _ in range(self.n_iters+1):
            yield np.array(sorted(random.sample(range(self.data_len), self.bsize)))

    def __len__(self):
        return self.n_iters+1

def get_dataloader(cfg):
	dataset = SPMLDataset(cfg, True)
	sampler = RandomSampler(cfg.DATA.bsize, len(dataset), cfg.DATA.max_iter)
	pose_metadata = dataset.get_meta()
	dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=24)
	return dataloader, pose_metadata
