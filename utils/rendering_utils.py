import numpy as np
import torch

def latlong_vec(res, device=None):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    dir_vec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    # return dr.texture(cubemap[None, ...], dir_vec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
    return dir_vec


def envmap_vec(res, device=None):
    return -latlong_vec(res, device).flip(0).flip(1)


def rgb2srgb(rgb):
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)


def reinhard(x, max_point=16):
    # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    # lumi = lumi[..., None]
    # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
    # y_rein = x / (1+x)
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein
