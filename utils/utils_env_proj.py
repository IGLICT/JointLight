import numpy as np
import torch
from . import rendering_utils as util

def rotate_envmap(envmap_world, yaw):
    latlong_img = envmap_world
    if isinstance(latlong_img, np.ndarray):
        latlong_img = torch.from_numpy(latlong_img)
    latlong_img = torch.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0)
    latlong_img = latlong_img.clamp(0.0, 65504.0)

    lat_h, lat_w = latlong_img.shape[:2]
    pixel_rot = int(lat_w * yaw / 360)
    latlong_img = torch.roll(latlong_img, shifts=pixel_rot, dims=1)

    return latlong_img


def hdr_mapping(env_hdr, log_scale=10000):
    """Map HDR environment maps to LDR and logarithmic representations."""
    env_ldr = util.rgb2srgb(util.reinhard(env_hdr, max_point=16).clamp(0, 1))
    env_log = util.rgb2srgb(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
    env_norm = util.rgb2srgb(torch.log1p(env_hdr) / torch.log1p(env_hdr).max())
    # env_norm = util.rgb2srgb(torch.log1p(env_hdr) / torch.log1p(env_hdr).max())
    return {
        'env_hdr': env_hdr,    # Original HDR image
        'env_ldr': env_ldr,    # LDR image after tone mapping
        'env_log': env_log,    # Logarithmic scaling
        'env_norm': env_norm,    # uniform scaling
    }
