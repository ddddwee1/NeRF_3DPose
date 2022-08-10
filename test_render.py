import yaml
from tqdm import tqdm
from easydict import EasyDict
from lib import dataloader, trainer
import numpy as np 
import cv2

if __name__=='__main__':
	cfg = EasyDict(yaml.load(open('config.yaml'), Loader=yaml.FullLoader))
	dataset = dataloader.SPMLDataset(cfg, False)
	batch = dataset[0]

	canvas = np.zeros([1000*1000, 3], dtype=np.float32)
	sampled_idx = batch['sampled_idx']
	rgb = batch['RGB']
	print(rgb.max(), rgb.min())
	print(rgb.shape, canvas[sampled_idx].shape)
	canvas[sampled_idx] = rgb

	canvas = np.uint8(canvas * 255).reshape([1000, 1000, 3])
	print(canvas.shape)
	cv2.imwrite('aa.png', canvas)
