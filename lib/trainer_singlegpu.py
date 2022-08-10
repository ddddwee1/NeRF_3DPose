import os
import torch
import random
import numpy as np 
import TorchSUL.Model as M

from lib import encoder, embedder
from lib.models import poseopt, nerf
from lib.utils import nerf_utils

class Trainer():
    def __init__(self, cfg, pose_metadata):
        self.cfg = cfg
        self.num_pts = pose_metadata['rest_joints'].shape[-2]
        self.pose_opt_layer = poseopt.PoseOptim(cfg, pose_metadata)
        
        self.encoder = encoder.Encoder()
        self.embedder = embedder.BasicEmbedder(cfg)

        self.net_coarse = nerf.NeRF(cfg)
        self.net_fine = nerf.NeRF(cfg)
        
    def initialize(self):
        dumb_u = self.embedder(torch.ones(1, self.num_pts*3))
        dumb_n = self.embedder(torch.ones(1, self.num_pts*1))
        dumb_v = self.embedder(torch.ones(1, self.num_pts*3))
        pose_embed = torch.cat([dumb_u, dumb_n], dim=-1)
        self.net_coarse(pose_embed, dumb_v)
        self.net_fine(pose_embed, dumb_v)

        self.pose_saver = M.Saver(self.pose_opt_layer)
        self.net_coarse_saver = M.Saver(self.net_coarse)
        self.net_fine_saver = M.Saver(self.net_fine)

        self.net_coarse_saver.restore(os.path.join(self.cfg.EXPERIMENT.name, 'net_coarse/'))
        self.net_fine_saver.restore(os.path.join(self.cfg.EXPERIMENT.name, 'net_fine/'))
        self.pose_saver.restore(os.path.join(self.cfg.EXPERIMENT.name, 'pose_model/'))

        self.pose_optimizer = torch.optim.SGD(self.pose_opt_layer.parameters(), lr=self.cfg.POSEOPT.lr)
        self.net_optimizer = torch.optim.Adam([{'params': self.net_coarse.parameters()}, {'params': self.net_fine.parameters()}], lr=self.cfg.MODEL.lr)

        self.net_coarse.cuda()
        self.net_fine.cuda()

        self.pose_opt_layer.cuda()
        self.embedder.cuda()

    def save(self, global_step):
        stamp = random.randint(1, 100000)
        fname = '%06d_%06d.pth'%(global_step, stamp)
        self.net_coarse_saver.save(os.path.join(self.cfg.EXPERIMENT.name, 'net_coarse', fname))
        self.net_fine_saver.save(os.path.join(self.cfg.EXPERIMENT.name, 'net_fine', fname))
        self.pose_saver.save(os.path.join(self.cfg.EXPERIMENT.name, 'pose_model', fname))

    def train_batch(self, batch, global_step):
        self.adjust_optimizer(global_step)
        rgb_coarse, rgb_fine = self.run_batch(batch, global_step)
        loss_coarse, loss_fine = self.img_loss(batch, rgb_coarse, rgb_fine)
        self.apply_loss([[loss_coarse, 1.0], [loss_fine, 1.0]])
        return loss_coarse, loss_fine

    # def adjust_optimizer(self, global_step):
    #     if global_step in self.cfg.TRAIN.n_iter_decay:
    #         for gp in self.pose_optimizer.param_groups:
    #             lr = gp['lr']
    #             lr = lr * 0.1 
    #             gp['lr'] = lr 
    #             print('Pose lr changed to:', lr)

    #         for gp in self.net_optimizer.param_groups:
    #             lr = gp['lr']
    #             lr = lr * 0.1 
    #             gp['lr'] = lr 
    #             print('Net lr changed to:', lr)

    def adjust_optimizer(self, global_step):
        if global_step%self.cfg.TRAIN.decay_interval==0:
            multiplier = np.exp(self.cfg.TRAIN.decay_gamma * global_step)

            for gp in self.pose_optimizer.param_groups:
                lr = self.cfg.POSEOPT.lr * multiplier
                gp['lr'] = lr 
                print('Pose lr changed to:', lr)

            for gp in self.net_optimizer.param_groups:
                lr = self.cfg.MODEL.lr * multiplier
                gp['lr'] = lr 
                print('Net lr changed to:', lr)

    def apply_loss(self, losses_weights):
        self.pose_optimizer.zero_grad()
        self.net_optimizer.zero_grad()

        ls_total = 0.0
        for ls,w in losses_weights:
            ls_total = ls_total + ls * w
        ls_total.backward()

        self.pose_optimizer.step()
        self.net_optimizer.step()

        # print(self.net_fine.alpha_fc.fc.weight.grad[0])
        # print(self.pose_opt_layer.poses.grad.max())

    def img_loss(self, batch, rgb_coarse, rgb_fine):
        # print(rgb_coarse.max(), rgb_coarse.min(), rgb_fine.max(), rgb_fine.min())
        rgb = batch['RGB'].to(rgb_coarse.device)
        loss_coarse = torch.abs(rgb_coarse - rgb).mean()
        loss_fine = torch.abs(rgb_fine - rgb).mean()
        return loss_coarse, loss_fine

    def run_batch(self, batch, global_step):
        joints, R = self.pose_opt_layer(batch['idx'])
        
        near = batch['near'].cuda()
        far = batch['far'].cuda()
        rays_o = batch['rays_o'].cuda()
        rays_d = batch['rays_d'].cuda()
        noise_std = self.cfg.TRAIN.noise_std * (self.cfg.TRAIN.noise_decay - global_step) / self.cfg.TRAIN.noise_decay
        if noise_std<0:
            noise_std = 0.0
        if global_step%1000==0:
            print('noise_std', noise_std)

        pts_coarse, z_vals_coarse = nerf_utils.sample_pts(rays_o, rays_d, near, far, self.cfg.MODEL.n_samples, perturb=True)
        pose_unit, pose_norm, view_local = self.encoder(pts_coarse, rays_d, joints, R)
        pose_unit = self.embedder(pose_unit)
        pose_norm = self.embedder(pose_norm)
        view_local = self.embedder(view_local)

        pose_embed = torch.cat([pose_unit, pose_norm], dim=-1)
        raw_coarse = self.net_coarse(pose_embed, view_local)
        rgb_coarse, weight_coarse = self.net_coarse.raw2outputs(raw_coarse, z_vals_coarse, rays_d, noise_std)

        pts_fine, z_vals_fine = nerf_utils.sample_fine(z_vals_coarse, weight_coarse, rays_o, rays_d, self.cfg.MODEL.n_importance)
        pose_unit, pose_norm, view_local = self.encoder(pts_fine, rays_d, joints, R)
        pose_unit = self.embedder(pose_unit)
        pose_norm = self.embedder(pose_norm)
        view_local = self.embedder(view_local)

        pose_embed = torch.cat([pose_unit, pose_norm], dim=-1)
        raw_fine = self.net_fine(pose_embed, view_local)
        rgb_fine, weight_fine = self.net_fine.raw2outputs(raw_fine, z_vals_fine, rays_d, noise_std)
        return rgb_coarse, rgb_fine
