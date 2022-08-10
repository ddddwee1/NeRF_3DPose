import torch

def sample_pts(rays_o, rays_d, near, far, N_samples, perturb=False):
    t_vals = torch.linspace(0., 1., N_samples).to(near.device)                                  # [S]
    z_vals = near[:, None] * (1 - t_vals) + far[:, None] * t_vals                               # [N, S]

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:,:-1])
        upper = torch.cat([mids, z_vals[:,-1:]], -1)
        lower = torch.cat([z_vals[:,:1], mids], -1)
        rand = torch.rand(z_vals.shape).to(z_vals.device) * (upper - lower)
        z_vals = lower + rand

    z_vals = z_vals.unsqueeze(1).expand(-1, rays_d.shape[1], -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals.unsqueeze(-1)
    return pts, z_vals

def sample_cdf(z_vals, weights, N_samples, det=False):
    bins = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
    weights = weights + 1e-5 
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., N_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1])+[N_samples])
    else:
        u = torch.rand(*cdf.shape[0:-1], N_samples)
    u = u.cuda(cdf.device)
    
    idxs = torch.searchsorted(cdf, u, right=True)
    below = torch.maximum(torch.zeros_like(idxs), idxs-1).long()
    above = torch.minimum(torch.ones_like(idxs)*(cdf.shape[-1]-1), idxs).long()

    # idxs_g = torch.stack([below, above], -1)
    cdf_below = torch.gather(cdf, dim=-1, index=below)
    cdf_above = torch.gather(cdf, dim=-1, index=above)

    bin_below = torch.gather(bins, dim=-1, index=below)
    bin_above = torch.gather(bins, dim=-1, index=above)

    denom = cdf_above - cdf_below
    denom = torch.clamp(denom, 1e-5, 99999)
    t = (u - cdf_below) / denom
    samples = bin_below + t * (bin_above - bin_below)
    return samples 

def sample_fine(z_vals, weights, rays_o, rays_d, N_samples, determinate=False):
    # z_vals = z_vals.unsqueeze(-2).broadcast_to(*weights.shape)
    cdf_samples = sample_cdf(z_vals, weights[..., 1:-1], N_samples, det=determinate).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, cdf_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals
    