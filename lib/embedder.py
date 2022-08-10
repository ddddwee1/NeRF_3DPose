import torch
import TorchSUL.Model as M

class BasicEmbedder(M.Model):
    def initialize(self, cfg):
        self.cfg = cfg
        self.periodic_fn = [torch.sin, torch.cos]
        self.register_buffer('freq_bands', 2. ** torch.linspace(0., cfg.n_freqs - 1 , cfg.n_freqs))            # [f]

    def forward(self, x):
        # x: any shape
        outs = [x]
        for fn in self.periodic_fn:
            res = fn(x[..., None] * self.freq_bands)
            res = res.reshape(*x.shape[:-1], x.shape[-1] * self.cfg.n_freqs)
            outs.append(res)
        outs = torch.cat(outs, dim=-1)
        return outs

class CutoffEmbedder(M.Model):
    def initialize(self, cfg):
        self.cfg = cfg
        if cfg.cutoff_dist>0:
            self.register_buffer('tau', torch.tensor(cfg.init_tau))
            self.register_buffer('cutoff_dist', torch.ones(cfg.n_joints)*cfg.cutoff_dist)                           # [J]
            self.scale_factor = torch.nn.Parameter(torch.tensor(cfg.init_scale_factor))
        else:
            print('NO CUTOFF')
        if cfg.n_freqs>0:
            self.register_buffer('freq_bands', 2. ** torch.linspace(0., cfg.n_freqs - 1 , cfg.n_freqs))             # [f]
        else:
            print('NO FREQ EMBED')
        self.periodic_fn = [torch.sin, torch.cos]
        
    def forward(self, x, dist):
        # x: [N, Ray, sample, J, *]
        # dist: [N, Ray, sample, J, 1]
        if self.cfg.cutoff_dist>0:
            assert dist.shape[-1] == 1, 'distance must be a scalar'
            dist = dist.detach()
            # dist = dist - (1 + torch.tanh(self.scale_factor)) * self.cutoff_dist[..., None]
            dist = dist - self.cutoff_dist[..., None]
            w = torch.sigmoid(- self.tau * dist)
        else:
            w = torch.ones(1, device=x.device)
        
        outs = [(x * w).flatten(start_dim=-2)]
        if self.cfg.n_freqs>0:
            for fn in self.periodic_fn:
                res = fn(x[..., None] * self.freq_bands)                    # [N, Ray, sample, J, *, f]
                res = res * w[..., None]
                res = res.flatten(start_dim=-3)                             # [N, Ray, sample, -1]
                outs.append(res)
        outs = torch.cat(outs, dim=-1)
        return outs

    def update_tau(self):
        if self.cfg.cutoff_dist>0:
            self.tau *= self.cfg.tau_amplifier
            self.tau.clamp_(self.cfg.init_tau, self.cfg.max_tau)
            print('EMBEDDER: TAU UPDATED:', self.tau)
            return self.tau

