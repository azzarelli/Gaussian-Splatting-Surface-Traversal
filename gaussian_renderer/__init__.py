import torch
import torch.nn.functional as F

from gsplat.rendering import rasterization

from utils.sh_utils import eval_sh

import time
def process_Gaussians(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    
    opacity = pc.get_opacity

    scales = pc.get_scaling #pc.get_scaling_with_3D_filter
    
    rotations = pc.rotation_activation(pc.splats["quats"])
    
    return means3D, rotations, opacity, colors, scales

def process_full_Gaussians(pc):
    # Use existing function for processing canon
    means3D, rotations, opacity, colors, scales = process_Gaussians(pc)
    
    invariance = pc.get_lambda
    texsample = pc.get_ab
    texscale = pc.get_texscale
        
    return means3D, rotations, opacity, colors, scales, texsample, texscale, invariance

def rendering_pass(means3D, rotation, scales, opacity, colors, cam, sh_deg=3, mode="RGB+D"):
    if mode in ['normals', '2D']:
        gmode = 'RGB+D'
    else:
        gmode = mode
    
    intr = cam.intrinsics.unsqueeze(0)
    viewmat = cam.w2c.unsqueeze(0)
    w2c = viewmat
    width = cam.image_width
    height = cam.image_height
    # Typical RGB render of base color
    colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        w2c.cuda(), 
        intr.cuda(),
        width, 
        height,
        
        render_mode=gmode,
        
        # rasterize_mode='antialiased',
        # eps2d=0.3,
        
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        sh_degree=sh_deg, #pc.active_sh_degree,
    )
        
    return colors, alphas, meta
import torch.nn.functional as F

def apply_colormap(render, threshold=0.0001):
    render = render.squeeze(-1)  # (H, W)
    
    mask = render > threshold  # (H, W)
    
    # Normalize to [0, 1]
    # render = (render - render.min()) / (render.max() - render.min() + 1e-8)
    
    # Red -> Blue
    r = 1.0 - render
    g = torch.zeros_like(render)
    b = render
    
    rgb = torch.stack([r, g, b], dim=0)  # (3, H, W)
    
    # Zero out background pixels
    rgb = rgb * mask.unsqueeze(0)
    
    return rgb

@torch.no_grad
def render(viewpoint_camera, pc, obj_pc, view_args=None):
    """
    Render the scene for viewing
    """
    
    # Main Gaussian Model    
    active_sh = 3
    means, rotations, opacity, colors, scales = process_Gaussians(pc)


    # Set arguments depending on type of viewing
    if view_args['vis_mode'] in 'render':
        mode = "RGB"
    elif view_args['vis_mode'] == 'D':
        mode = "D"
    elif view_args['vis_mode'] == 'xyz':
        colors = means.unsqueeze(0)
        active_sh=None
        mode = "RGB"
    else:
        mode = "RGB"

    # Render
    render, alpha, _ = rendering_pass(
        means, rotations, scales, opacity, colors,
        viewpoint_camera, 
        active_sh,
        mode=mode
    )
    

    # Process image
    if view_args['vis_mode'] == 'render':
        render = render.squeeze(0).permute(2,0,1)

    elif view_args['vis_mode'] == 'D':
        render = (render - render.min())/ (render.max() - render.min())
        render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
    
    elif view_args['vis_mode'] == 'xyz':
        render = render.squeeze(0).permute(2,0,1)

    
    # Overlay the object
    if obj_pc.splats != None:
        means, rotations, opacity, colors, scales = obj_pc.process_Gaussians()
        render_obj, alpha_obj, _ = rendering_pass(
            means, rotations, scales, opacity, colors,
            viewpoint_camera, 
            3,
            mode='RGB'
        )
        render_obj = render_obj.squeeze(0).permute(2,0,1)
        alpha_obj = alpha_obj.squeeze(0).permute(2,0,1)
        
        render = render_obj*alpha_obj + (1.-alpha_obj)*render

    return render


import torch.nn.functional as F
def generate_mipmaps(I, num_levels=3):
    I = I.unsqueeze(0)
    maps = [I]    
    for _ in range(1, num_levels):
        # I progressively downsampled
        I = F.interpolate(
            I, scale_factor=0.5,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )
        # Add zero padding
        I_ = I
        maps.append(I_)
    return maps

def sample_mipmap(I, uv, s, num_levels=3):
    """
    args:
        uv, Tensor, N,2
        s, Tensor, N,1
        I, Tensor, 3, H, W
    """
    N = s.size(0)
    
    # print(I.shape)
    # exit()
    # 1. Generate mipmaps
    maps = generate_mipmaps(I, num_levels=num_levels)
    
    # Normalize us -1, 1 (from 0, 1)
    uv = 2.*uv -1.
    uv = uv.unsqueeze(0).unsqueeze(0) # for grid_sample input we need, N,Hout,Wout,2, where N =1, and W=number of points
    
    # Scaling mip-maps
    L = s*(num_levels-1.)
    lower = torch.floor(L).long().clamp(max=num_levels-1)
    upper = torch.clamp(lower + 1, max=num_levels-1)
    s_interp = (L - lower.float())

    # Initialize mipmap samples
    mip_samples = torch.empty((N, num_levels, 3), device=s.device)    

    # For each map sample using u,v and store the values in samples
    for idx, map in enumerate(maps):
        # map is (1, 3, h, w)
        mip_samples[:, idx] = F.grid_sample(map, uv, mode='bilinear', align_corners=False, padding_mode='border').squeeze(2).squeeze(0).permute(1,0)

    gather_idx_low  = lower.view(N, 1, 1).expand(-1, 1, 3)
    gather_idx_high = upper.view(N, 1, 1).expand(-1, 1, 3)
    colors_low  = torch.gather(mip_samples, 1, gather_idx_low).squeeze(1)   # [N,3]
    colors_high = torch.gather(mip_samples, 1, gather_idx_high).squeeze(1) 
    
    colors = (1. - s_interp) * colors_low + s_interp * colors_high

    return colors


def get_colors_from_xyz(means3D):
    colors = means3D.unsqueeze(0)

def render_draw_mouse_click(viewpoint_camera, pc, x, y):
    """
    Render the scene for viewing
    """
    extras = None


    means, rotation, opacity, _, scales = process_Gaussians(pc)
    
    # Get XYZ val
    
    colors = torch.cat([means, scales, rotation], dim=-1).unsqueeze(0)
    render, _, _ = rendering_pass(
        means, rotation, scales, opacity, colors,
        viewpoint_camera, 
        None,
        mode='RGB'
    )
    res = render[0, y, x]
    
    return res[:3], res[3:6], res[6:]
    # return mean, scale, quat


    
