import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchSUL import Model as M

class NeRF(M.Model):
    def initialize(self, cfg, D=8, W=256, skips=[4], n_camcodes=None, n_camchannel=16):
        self.cfg = cfg
        # use_viewdirs = True 
        self.layers = nn.ModuleList()
        for i in range(D):
            self.layers.append(M.Dense(W, activation=M.PARAM_RELU))
        
        self.skips = skips

        self.alpha_fc = M.Dense(1)
        self.bottleneck = M.Dense(256)
        self.hidden = M.Dense(W//2)
        self.out_fc = M.Dense(3)
        
        self.n_camcodes = n_camcodes
        print('N_framecodes:', self.n_camcodes)
        if n_camcodes is not None:
            self.cam_embedder = nn.Embedding(n_camcodes, n_camchannel)

    def forward(self, embed, embeddir, cam_idx=None):
        x = embed 
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i in self.skips:
                x = torch.cat([embed, x], dim=-1)
        
        alpha_out = self.alpha_fc(x)

        bottleneck = self.bottleneck(x)
        if self.n_camcodes is not None:
            cam_codes = self.cam_embedder(cam_idx.long()).squeeze(dim=-2)
            embeddir = torch.cat([embeddir, cam_codes], -1)
        x = torch.cat([bottleneck, embeddir],dim=-1)
        out = self.out_fc(x)
        out = torch.cat([out, alpha_out], -1)
        return out
    
    def raw2alpha(self, raw, dists, noise):
        return 1.0 - torch.exp(- F.relu(raw + noise) * dists)
    
    def raw2outputs(self, raw, z_vals, rays_d, noise_std=0.0):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[...,:1])*1e10], -1)
        # dists = dists.unsqueeze(1)                                                      # [N, 1, sample]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)                          # [N, rays, sample]
        rgb = torch.sigmoid(raw[...,:3])
        if noise_std>0:
            noise = torch.randn(raw[...,3].shape, device=raw.device) * noise_std
        else:
            noise = 0
        alpha = self.raw2alpha(raw[...,3], dists, noise)
        # exclusive cumprod 
        weights = 1. - alpha + 1e-10
        weights = torch.cat([torch.ones_like(weights[...,:1]), weights[...,:-1]], dim=-1)
        weights = alpha * torch.cumprod( weights, dim=-1)
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)
        return rgb_map, weights

