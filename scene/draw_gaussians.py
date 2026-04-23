import torch

from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.base_gaussian import BasicGaussianModel
class ObjectModel(BasicGaussianModel):

    def __init__(self,):
        super().__init__()

    def process_Gaussians(self):
        means3D = self.splats['means']
        colors = self.get_features
        
        opacity = self.splats['opacities']

        scales = self.splats['scales'] 
        
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
        self.splats["sh0"] = torch.cat([self.splats["sh0"], torch.tensor([[[1., 1., 1.]]]).float().cuda()], dim=0)
        self.splats["shN"] = torch.cat([self.splats["shN"], torch.zeros((1, 15, 3)).float().cuda()], dim=0)
        
        
        # Connect two Gaussians
        if self.point_count > 1:
            A = self.get_xyz[-2]
            B = self.get_xyz[-1]
            
            
            num_samples = 10
            for j in range(1, num_samples-1):
                
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
                ray_o, ray_d = generate_perpendiculat_rays(mid, dir)
                
                intersections = []
                for i in range(16):
                    dist = find_surface_intersection(
                        gaussians,
                        ray_origin=ray_o[i],
                        ray_dir=ray_d[i],
                        sample_indices=sample_indices,
                        max_dist=max_dist,
                        num_steps=200,
                        threshold=0.5,
                    )
                    intersections.append(dist)
                            
                    if debug:
                        debug_point = ray_o[i] + max_dist * ray_d[i]
                        self.splats["means"] = torch.cat([self.splats["means"], debug_point.unsqueeze(0)], dim=0)
                        self.splats["scales"] = torch.cat([self.splats["scales"], scale.unsqueeze(0)], dim=0)
                        self.splats["quats"] = torch.cat([self.splats["quats"], quat.unsqueeze(0)], dim=0)
                        self.splats["opacities"] = torch.cat([self.splats["opacities"], torch.tensor([[1.]]).float().cuda()], dim=0)
                        self.splats["sh0"] = torch.cat([self.splats["sh0"], torch.tensor([[[1., 1., 0.]]]).float().cuda()], dim=0)  # green
                        self.splats["shN"] = torch.cat([self.splats["shN"], torch.zeros((1, 15, 3)).float().cuda()], dim=0)


                valid = [(d, i) for i, d in enumerate(intersections) if d is not None]
                if not valid:
                    print("No intersections found! Try lowering threshold or increasing max_dist")
                    return

                min_dist, min_idx = min(valid, key=lambda x: x[0])
                intersection_point = ray_o[min_idx] + (min_dist*1.5) * ray_d[min_idx]
                
                if debug:
                    selected_splat_idx = -(16 - min_idx)
                    self.splats["sh0"][selected_splat_idx] = torch.tensor([[1., 0., 0.]]).float().cuda()

                self.splats["means"] = torch.cat([self.splats["means"], intersection_point.unsqueeze(0)], dim=0)
                self.splats["scales"] = torch.cat([self.splats["scales"], scale.unsqueeze(0)], dim=0)
                self.splats["quats"] = torch.cat([self.splats["quats"], quat.unsqueeze(0)], dim=0)
                self.splats["opacities"] = torch.cat([self.splats["opacities"], torch.tensor([[1.]]).float().cuda()], dim=0)
                self.splats["sh0"] = torch.cat([self.splats["sh0"], torch.tensor([[[0., 1., 0.]]]).float().cuda()], dim=0)
                self.splats["shN"] = torch.cat([self.splats["shN"], torch.zeros((1, 15, 3)).float().cuda()], dim=0)
                
                A = mid
            
    
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
    rots = gaussians.get_rotation[sample_indices]       # (M, 4)

    if xyz.shape[0] == 0:
        return torch.tensor(0.0, device=point.device)

    # Rotate diff into local Gaussian frame using quaternion transpose (no inversion needed)
    w, x, y, z = rots.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        torch.stack([1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)], dim=-1),
        torch.stack([2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)], dim=-1),
    ], dim=-2)  # (M, 3, 3)

    diff = (point.unsqueeze(0) - xyz).unsqueeze(-1)                    # (M, 3, 1)
    diff_local = torch.bmm(R.transpose(1, 2), diff).squeeze(-1)        # (M, 3)
    mahal = ((diff_local / scales) ** 2).sum(dim=-1)                   # (M,)

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
