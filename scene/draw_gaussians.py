import torch

from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.base_gaussian import BasicGaussianModel



class ObjectModel(BasicGaussianModel):

    def __init__(self,):
        super().__init__()
        
        self.base_color = [1., 1., 1.]

    def process_Gaussians(self, scale_adjust=None):
        means3D = self.splats['means']
        colors = self.get_features
        
        opacity = self.splats['opacities']

        scales = self.splats['scales'] if scale_adjust == None else self.splats["scales"] * 0. + scale_adjust 

        rotations = self.splats["quats"]
        
        return means3D, rotations, opacity, colors, scales

    def add_gaussian(self,mean, scale, quat, gaussians, debug=False):
        if self.splats == None:
            self.splats = {
                "means":torch.empty((0, 3)).cuda().float(),
                "scales":torch.empty((0, 3)).cuda().float(),
                "quats":torch.empty((0, 4)).cuda().float(),
                "opacities":torch.empty((0, 1)).cuda().float(),
                "sh0":torch.empty((0, 1, 3)).cuda().float(),
                "shN":torch.empty((0, 15, 3)).cuda().float()
            }
        
        self.splats["means"] = torch.cat([self.splats["means"], mean.unsqueeze(0)], dim=0)
        self.splats["scales"] = torch.cat([self.splats["scales"], scale.unsqueeze(0)], dim=0)
        self.splats["quats"] = torch.cat([self.splats["quats"], quat.unsqueeze(0)], dim=0)
        self.splats["opacities"] = torch.cat([self.splats["opacities"], torch.tensor([[1.]]).float().cuda()], dim=0)
        self.splats["sh0"] = torch.cat([self.splats["sh0"], torch.tensor([[self.base_color]]).float().cuda()], dim=0)
        self.splats["shN"] = torch.cat([self.splats["shN"], torch.zeros((1, 15, 3)).float().cuda()], dim=0)
        
        
        # Connect two Gaussians
        if self.point_count > 1:
            A = self.get_xyz[-2]
            B = self.get_xyz[-1]
            
            
            num_samples = 10
            for j in range(1, num_samples):
                
                dir = B - A
                t = (j)/num_samples
                
                mid = A + t * dir
                dir = dir / torch.norm(dir)
                
                ab_dist = torch.norm(B - A).item()
                max_dist = ab_dist * 1.0
                
                # To reduce computation:
                #    - Ball-point query to sample only proximal gaussians
                sample_indices = ((mid - gaussians.get_xyz).pow(2)).sum(dim=-1).sqrt()
                sample_indices = (sample_indices < max_dist)
                
                # Construct the radial plane basis around AB @ mid
                ray_o, ray_d = generate_perpendicular_rays(mid, dir)

                hit_dists = find_surface_intersections_batched(
                    gaussians, ray_o, ray_d,
                    sample_indices=sample_indices,
                    max_dist=max_dist,
                    num_steps=200,
                    threshold=0.5,
                )

                finite = torch.isfinite(hit_dists)
                if not finite.any():
                    print("No intersections found! Try lowering threshold or increasing max_dist")
                    return

                min_idx = torch.where(finite, hit_dists, torch.full_like(hit_dists, float('inf'))).argmin()
                min_dist = hit_dists[min_idx]
                intersection_point = ray_o[min_idx] + (min_dist * 1.5) * ray_d[min_idx]
                
                if debug:
                    selected_splat_idx = -(16 - min_idx)
                    self.splats["sh0"][selected_splat_idx] = torch.tensor([[1., 0., 0.]]).float().cuda()

                self.splats["means"] = torch.cat([self.splats["means"], intersection_point.unsqueeze(0)], dim=0)
                self.splats["scales"] = torch.cat([self.splats["scales"], scale.unsqueeze(0)], dim=0)
                self.splats["quats"] = torch.cat([self.splats["quats"], quat.unsqueeze(0)], dim=0)
                self.splats["opacities"] = torch.cat([self.splats["opacities"], torch.tensor([[1.]]).float().cuda()], dim=0)
                self.splats["sh0"] = torch.cat([self.splats["sh0"], torch.tensor([[self.base_color]]).float().cuda()], dim=0)
                self.splats["shN"] = torch.cat([self.splats["shN"], torch.zeros((1, 15, 3)).float().cuda()], dim=0)
                            
    
    def reset(self):
        self.splats = {
            "means":torch.empty((0, 3)).cuda().float(),
            "scales":torch.empty((0, 3)).cuda().float(),
            "quats":torch.empty((0, 4)).cuda().float(),
            "opacities":torch.empty((0, 1)).cuda().float(),
            "sh0":torch.empty((0, 1, 3)).cuda().float(),
            "shN":torch.empty((0, 15, 3)).cuda().float()
        }
from scipy.spatial import KDTree
import torch

def generate_perpendiculat_rays(mid, dir):
    arbitrary = torch.tensor([1.0, 0.0, 0.0], device=dir.device)
    if torch.abs(torch.dot(dir, arbitrary)) > 0.99:  # too parallel, pick another
        arbitrary = torch.tensor([0.0, 1.0, 0.0], device=dir.device)
    
    u = torch.cross(dir, arbitrary, dim=-1)
    u = u / torch.norm(u)
    v = torch.cross(dir, u, dim=-1)
    v = v / torch.norm(v)

    # Sample 16 directions uniformly around the ring
    angles = torch.linspace(0, 2 * torch.pi, 17, device=dir.device)[:-1]  # drop last to avoid duplicate
    ray_dirs = torch.stack([
        torch.cos(a) * u + torch.sin(a) * v for a in angles
    ])  # shape: (16, 3)
    ray_origins = mid.unsqueeze(0).expand(16, -1)
    return ray_origins, ray_dirs


def sample_gaussian_density(gaussians, point, sample_indices):
    xyz = gaussians.get_xyz[sample_indices]
    opacities = gaussians.get_opacity.squeeze(-1)[sample_indices]
    scales = gaussians.get_scaling[sample_indices]      # (M, 3)
    inv_cov = gaussians.inv_covariance[sample_indices]     # (M, 4)

    if xyz.shape[0] == 0:
        return torch.tensor(0.0, device=point.device)

    diff = point.unsqueeze(0) - xyz                                    # (M, 3)
    mahal = torch.einsum('ni,nij,nj->n', diff, inv_cov, diff)          # (M,)
    return (opacities * torch.exp(-0.5 * mahal)).sum()

def find_surface_intersection(gaussians, ray_origin, ray_dir, sample_indices=None,
                              max_dist=5.0, num_steps=200, threshold=0.9):
    """
    March along a ray, accumulating density until cumulative alpha hits threshold.
    Returns distance of intersection.
    """
    step_size = max_dist / num_steps
    transmittance = 1.0
    
    for i in range(num_steps):
        t = i * step_size
        point = ray_origin + t * ray_dir

        density = sample_gaussian_density(gaussians, point, sample_indices)

        if density < threshold:
            # Refine: binary search between t-step_size and t
            t_lo = (i - 1) * step_size
            t_hi = t
            for _ in range(8):  # 8 refinement steps
                t_mid = 0.5 * (t_lo + t_hi)
                point_mid = ray_origin + t_mid * ray_dir
                d = sample_gaussian_density(gaussians, point_mid, sample_indices)
                if d < threshold:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            return t_hi
    
    return None  # no intersection found


def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)



def generate_perpendicular_rays(mid, dir, num_rays=16):
    # pick an arbitrary axis not parallel to dir
    arbitrary = torch.tensor([1.0, 0.0, 0.0], device=dir.device)
    if torch.abs(torch.dot(dir, arbitrary)) > 0.99:
        arbitrary = torch.tensor([0.0, 1.0, 0.0], device=dir.device)

    u = torch.cross(dir, arbitrary, dim=-1); u = u / torch.norm(u)
    v = torch.cross(dir, u, dim=-1);         v = v / torch.norm(v)

    angles = torch.linspace(0, 2 * torch.pi, num_rays + 1, device=dir.device)[:-1]
    ray_dirs = torch.cos(angles)[:, None] * u + torch.sin(angles)[:, None] * v  # (R, 3)
    ray_origins = mid.unsqueeze(0).expand(num_rays, -1).contiguous()           # (R, 3)
    return ray_origins, ray_dirs


def sample_density_batch(xyz_sel, opac_sel, inv_cov_sel, points):
    """
    Evaluate density at many points against a pre-selected subset of Gaussians.
    xyz_sel:    (M, 3)
    opac_sel:   (M,)
    inv_cov_sel:(M, 3, 3)
    points:     (..., 3)   arbitrary leading shape
    returns:    (...,)     density per point
    """
    lead_shape = points.shape[:-1]
    P = points.reshape(-1, 3)                        # (P, 3)
    diff = P[:, None, :] - xyz_sel[None, :, :]       # (P, M, 3)
    # (P,M,3) x (M,3,3) x (P,M,3)  -> (P,M)
    mahal = torch.einsum('pmi,mij,pmj->pm', diff, inv_cov_sel, diff)
    contrib = opac_sel[None, :] * torch.exp(-0.5 * mahal)  # (P, M)
    return contrib.sum(dim=-1).reshape(lead_shape)

def find_surface_intersections_batched(gaussians, ray_origins, ray_dirs,
                                       sample_indices, max_dist=5.0,
                                       num_steps=200, threshold=0.5,
                                       refine_iters=8, step_chunk=16):
    device = ray_origins.device
    R = ray_origins.shape[0]

    xyz_sel     = gaussians.get_xyz[sample_indices]
    opac_sel    = gaussians.get_opacity.squeeze(-1)[sample_indices]
    inv_cov_sel = gaussians.inv_covariance[sample_indices]

    if xyz_sel.shape[0] == 0:
        return torch.full((R,), float('inf'), device=device)

    ts = torch.linspace(0.0, max_dist, num_steps, device=device)

    # March in chunks; stop early once every ray has hit
    first_idx = torch.full((R,), -1, dtype=torch.long, device=device)
    for start in range(0, num_steps, step_chunk):
        end = min(start + step_chunk, num_steps)
        ts_chunk = ts[start:end]                                   # (Sc,)
        pts = ray_origins[None] + ts_chunk[:, None, None] * ray_dirs[None]  # (Sc, R, 3)
        dens = sample_density_batch(xyz_sel, opac_sel, inv_cov_sel, pts)    # (Sc, R)
        below = dens < threshold                                   # (Sc, R)

        # For rays that haven't hit yet, find first hit in this chunk
        not_hit_yet = first_idx < 0
        chunk_any = below.any(dim=0) & not_hit_yet
        if chunk_any.any():
            chunk_first = below.float().argmax(dim=0) + start      # global index
            first_idx = torch.where(chunk_any, chunk_first, first_idx)

        if (first_idx >= 0).all():
            break

    any_hit = first_idx >= 0
    safe_idx = first_idx.clamp(min=0)
    hi = ts[safe_idx]
    lo = ts[(safe_idx - 1).clamp(min=0)]

    for _ in range(refine_iters):
        mid_t = 0.5 * (lo + hi)
        mid_pts = ray_origins + mid_t[:, None] * ray_dirs
        d = sample_density_batch(xyz_sel, opac_sel, inv_cov_sel, mid_pts)
        miss = d < threshold
        hi = torch.where(miss, mid_t, hi)
        lo = torch.where(miss, lo, mid_t)

    return torch.where(any_hit, hi, torch.full_like(hi, float('inf')))