import os
import torch
import random
import numpy as np 
import TorchSUL.Model as M

from lib import encoder, embedder, vggloss
from lib.models import poseopt, nerf
from lib.utils import nerf_utils

class NeRF_DP(M.Model):
    def initialize(self, cfg, num_pts):
        self.cfg = cfg
        self.encoder = encoder.Encoder()
        self.embedder_pts = embedder.CutoffEmbedder(cfg.EMBED.PTS)
        self.embedder_dist = embedder.CutoffEmbedder(cfg.EMBED.DIST)
        self.embedder_view = embedder.CutoffEmbedder(cfg.EMBED.VIEW)

        self.net_coarse = nerf.NeRF(cfg)
        self.net_fine = nerf.NeRF(cfg)

        dumb_u = self.embedder_pts(torch.ones(1, num_pts, 3), torch.ones(1, num_pts, 1))
        dumb_n = self.embedder_dist(torch.ones(1, num_pts, 1), torch.ones(1, num_pts, 1))
        dumb_v = self.embedder_view(torch.ones(1, num_pts, 3), torch.ones(1, num_pts, 1))
        print('EMBEDDED:', dumb_u.shape, dumb_n.shape, dumb_v.shape)
        pose_embed = torch.cat([dumb_u, dumb_n], dim=-1)
        self.net_coarse(pose_embed, dumb_v)
        self.net_fine(pose_embed, dumb_v)

    def update_tau(self):
        self.embedder_pts.update_tau()
        self.embedder_dist.update_tau()
        self.embedder_view.update_tau()

    def forward(self, pts_coarse, rays_o, rays_d, joints, R, z_vals_coarse, noise_std, determinate=False):
        pose_unit, pose_norm, view_local = self.encoder(pts_coarse, rays_d, joints, R)
        embed_pose_unit = self.embedder_pts(pose_unit, pose_norm)
        embed_pose_norm = self.embedder_dist(pose_norm, pose_norm)
        embed_view_local = self.embedder_view(view_local, pose_norm)

        pose_embed = torch.cat([embed_pose_unit, embed_pose_norm], dim=-1)
        raw_coarse = self.net_coarse(pose_embed, embed_view_local)
        rgb_coarse, weight_coarse = self.net_coarse.raw2outputs(raw_coarse, z_vals_coarse, rays_d, noise_std)

        pts_fine, z_vals_fine = nerf_utils.sample_fine(z_vals_coarse, weight_coarse, rays_o, rays_d, self.cfg.MODEL.n_importance, determinate=determinate)
        pose_unit, pose_norm, view_local = self.encoder(pts_fine, rays_d, joints, R)
        embed_pose_unit = self.embedder_pts(pose_unit, pose_norm)
        embed_pose_norm = self.embedder_dist(pose_norm, pose_norm)
        embed_view_local = self.embedder_view(view_local, pose_norm)

        pose_embed = torch.cat([embed_pose_unit, embed_pose_norm], dim=-1)
        raw_fine = self.net_fine(pose_embed, embed_view_local)
        rgb_fine, weight_fine = self.net_fine.raw2outputs(raw_fine, z_vals_fine, rays_d, noise_std)
        return rgb_coarse, rgb_fine, weight_coarse, weight_fine, pts_coarse, pts_fine


class Trainer():
    def __init__(self, cfg, pose_metadata):
        self.cfg = cfg
        self.pose_opt_layer = poseopt.PoseOptim(cfg, pose_metadata)
        num_pts = pose_metadata['rest_joints'].shape[-2]
        self.nerf_dp = NeRF_DP(cfg, num_pts)
        self.VGG = vggloss.VGGLoss()

    def initialize(self):
        self.pose_saver = M.Saver(self.pose_opt_layer)
        self.net_saver = M.Saver(self.nerf_dp)

        self.net_saver.restore(os.path.join(self.cfg.EXPERIMENT.name, 'net_ckpt/'))
        self.pose_saver.restore(os.path.join(self.cfg.EXPERIMENT.name, 'pose_model/'))
        self.pose_opt_layer.cuda()
        self.nerf_dp = torch.nn.DataParallel(self.nerf_dp)
        self.nerf_dp.cuda()

        self.pose_optimizer = torch.optim.Adam(self.pose_opt_layer.parameters(), lr=self.cfg.POSEOPT.lr)
        self.net_optimizer = torch.optim.Adam(self.nerf_dp.parameters(), lr=self.cfg.MODEL.lr)
        
        x = torch.zeros(2, 3, 512, 512)
        self.VGG(x, x)
        self.VGG.cuda()

    def save(self, global_step):
        stamp = random.randint(1, 100000)
        fname = '%06d_%06d.pth'%(global_step, stamp)
        self.net_saver.save(os.path.join(self.cfg.EXPERIMENT.name, 'net_ckpt', fname))
        self.pose_saver.save(os.path.join(self.cfg.EXPERIMENT.name, 'pose_model', fname))

    def train_batch(self, batch, global_step):
        self.adjust_optimizer(global_step)
        rgb_coarse, rgb_fine, weight_coarse, weight_fine = self.run_batch(batch, global_step)
        loss_coarse, loss_fine = self.img_loss(batch, rgb_coarse, rgb_fine)
        loss_weight_coarse, loss_weight_fine = self.entropy_loss(weight_coarse, weight_fine)
        loss_vgg_coarse, loss_vgg_fine = self.vgg_loss(batch, rgb_coarse, rgb_fine)
        if global_step>self.cfg.TRAIN.entropy_start_iter:
            self.apply_loss([[loss_coarse, 1.0], \
                            [loss_fine, 1.0], \
                            [loss_weight_coarse, self.cfg.LOSS.coarse_weight_entropy], \
                            [loss_weight_fine, self.cfg.LOSS.fine_weight_entropy],\
                            [loss_vgg_coarse, self.cfg.LOSS.weight_vgg], \
                            [loss_vgg_fine, self.cfg.LOSS.weight_vgg]], global_step)
        else:
            self.apply_loss([[loss_coarse, 1.0], [loss_fine, 1.0], [loss_weight_coarse, 0.0], [loss_weight_fine, 0.0], [loss_vgg_coarse, 0.0], [loss_vgg_fine, 0.0]], global_step)
        if global_step%self.cfg.EMBED.tau_update_interval==0 and global_step>0:
            self.nerf_dp.module.update_tau()
        return loss_coarse, loss_fine, loss_weight_coarse, loss_weight_fine, loss_vgg_coarse, loss_vgg_fine

    def vgg_loss(self, batch, rgb_coarse, rgb_fine):
        psize = self.cfg.DATA.sample_patch_size
        rgb_coarse = rgb_coarse.reshape(-1, psize, psize, 3)
        rgb_coarse = torch.permute(rgb_coarse, [0, 3, 1, 2])
        rgb_fine = rgb_fine.reshape(-1, psize, psize, 3)
        rgb_fine = torch.permute(rgb_fine, [0, 3, 1, 2])
        rgb = batch['RGB'].to(rgb_coarse.device).reshape(-1, psize, psize, 3)
        rgb = torch.permute(rgb, [0, 3, 1, 2])
        mask = batch['training_mask'].to(rgb_coarse.device).reshape(-1, 1, psize, psize)
        loss_coarse = self.VGG(rgb_coarse*mask, rgb*mask)
        loss_fine = self.VGG(rgb_fine*mask, rgb*mask)
        return loss_coarse, loss_fine

    def img_loss(self, batch, rgb_coarse, rgb_fine):
        # print(rgb_coarse.max(), rgb_coarse.min(), rgb_fine.max(), rgb_fine.min())
        rgb = batch['RGB'].to(rgb_coarse.device)
        mask = batch['training_mask'].to(rgb_coarse.device).unsqueeze(-1)
        loss_coarse = torch.pow(rgb_coarse*mask - rgb*mask, 2).mean()
        loss_fine = torch.pow(rgb_fine*mask - rgb*mask, 2).mean()
        return loss_coarse, loss_fine

    def _entropy(self, x):
        weight = x[...,:-1].sum(dim=-1)
        idx = torch.where(weight>0.9)
        if len(idx[0])==0:
            return torch.tensor(0.0).to(x.device)
        sampled_selected = x[idx[0], idx[1]]            # [N, samples]
        sampled_selected = sampled_selected[:,:-1]      # [N, samples-1]
        ent = -(sampled_selected * torch.log(sampled_selected+1e-6)).sum(dim=-1).mean()
        return ent

    def entropy_loss(self, weight_coarse, weight_fine):
        # weight: [N, Rays, num_samples]
        loss_weight_coarse = self._entropy(weight_coarse)
        loss_weight_fine = self._entropy(weight_fine)
        return loss_weight_coarse, loss_weight_fine

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
                # print('Pose lr changed to:', lr)

            for gp in self.net_optimizer.param_groups:
                lr = self.cfg.MODEL.lr * multiplier
                gp['lr'] = lr 
                # print('Net lr changed to:', lr)

    def apply_loss(self, losses_weights, global_step):
        self.net_optimizer.zero_grad()

        ls_total = 0.0
        for ls,w in losses_weights:
            ls_total = ls_total + ls * w
        ls_total.backward()

        self.net_optimizer.step()

        if global_step%self.cfg.POSEOPT.update_iter==0 and global_step>0:
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()

    def run_batch(self, batch, global_step):
        joints, R, near, far = self.pose_opt_layer(batch['idx'])
        
        # near = batch['near'].cuda()
        # far = batch['far'].cuda()
        rays_o = batch['rays_o'].cuda()
        rays_d = batch['rays_d'].cuda()
        noise_std = self.cfg.TRAIN.noise_std * (self.cfg.TRAIN.noise_decay - global_step) / self.cfg.TRAIN.noise_decay
        if noise_std<0:
            noise_std = 0.0
        # if global_step%1000==0:
        #     print('noise_std', noise_std)

        pts_coarse, z_vals_coarse = nerf_utils.sample_pts(rays_o, rays_d, near, far, self.cfg.MODEL.n_samples, perturb=True)
        rgb_coarse, rgb_fine, weight_coarse, weight_fine, _, _ = self.nerf_dp(pts_coarse, rays_o, rays_d, joints, R, z_vals_coarse, noise_std)
        return rgb_coarse, rgb_fine, weight_coarse, weight_fine

    def evaluate(self, data):
        joints, R, near, far = self.pose_opt_layer([data['idx']])
        
        rays_o = data['rays_o'].unsqueeze(1).cuda()
        rays_d = data['rays_d'].unsqueeze(1).cuda()
        
        size = rays_o.shape[0]
        near = near.expand(size).cuda()
        far = far.expand(size).cuda()
        
        joints = joints.expand(size, -1, -1)
        R = R.expand(size, -1, -1, -1)
        noise_std = 0.0
        
        pts_coarse, z_vals_coarse = nerf_utils.sample_pts(rays_o, rays_d, near, far, self.cfg.MODEL.n_samples, perturb=False)
        rgbs = []
        i = 0
        while 1:
            start = i 
            end = i + self.cfg.TEST.chunk * torch.cuda.device_count()
            if pts_coarse.shape[0]-end<self.cfg.TEST.chunk:
                end = pts_coarse.shape[0]
            rgb_coarse, rgb_fine, _, _, _, _ = self.nerf_dp(pts_coarse[start:end], rays_o[start:end], rays_d[start:end], joints[start:end], R[start:end], z_vals_coarse[start:end], noise_std, determinate=True)
            rgbs.append(rgb_fine)
            if end==pts_coarse.shape[0]:
                break
            else:
                i = end
        rgbs = torch.cat(rgbs, dim=0).squeeze(1)
        return rgbs
        
    def evaluate_depth(self, data):
        joints, R, near, far = self.pose_opt_layer([data['idx']])
        
        rays_o = data['rays_o'].unsqueeze(1).cuda()
        rays_d = data['rays_d'].unsqueeze(1).cuda()
        
        size = rays_o.shape[0]
        near = near.expand(size).cuda()
        far = far.expand(size).cuda()
        
        joints = joints.expand(size, -1, -1)
        R = R.expand(size, -1, -1, -1)
        noise_std = 0.0
        
        pts_coarse, z_vals_coarse = nerf_utils.sample_pts(rays_o, rays_d, near, far, self.cfg.MODEL.n_samples, perturb=False)
        zvals = []
        weights = []
        
        i = 0
        while 1:
            start = i 
            end = i + self.cfg.TEST.chunk * torch.cuda.device_count()
            if pts_coarse.shape[0]-end<self.cfg.TEST.chunk:
                end = pts_coarse.shape[0]
            rgb_coarse, rgb_fine, weight_coarse, weight_fine, z_coarse, z_fine = self.nerf_dp(pts_coarse[start:end], rays_o[start:end], rays_d[start:end], joints[start:end], R[start:end], z_vals_coarse[start:end], noise_std, determinate=True)
            weights.append(weight_fine)
            zvals.append(z_fine)
            if end==pts_coarse.shape[0]:
                break
            else:
                i = end
        weights = torch.cat(weights, dim=0).squeeze(1)
        zvals = torch.cat(zvals, dim=0).squeeze(1)
        return weights, zvals
        
