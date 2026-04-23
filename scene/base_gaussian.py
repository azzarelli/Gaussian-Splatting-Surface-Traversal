import torch
import numpy as np
import os

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

from utils.general_utils import strip_symmetric, build_scaling_rotation

from plyfile import PlyData, PlyElement


class BasicGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.sigmoid_activation = torch.sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self,):
        self.active_sh_degree = 3        
        self.setup_functions()
        
        self.splats = None

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
                        
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.1
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.fx + camera.image_width / 2.0
            y = y / z * camera.fy + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.fx:
                focal_length = camera.fx
        
        distance[~valid_points] = distance[valid_points].max()
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        # self.filter_3D = filter_3D[..., None]

    @property
    def get_features(self):
        features_dc = self.splats['sh0']
        features_rest = self.splats['shN']
        return torch.cat((features_dc, features_rest), dim=1)
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self.splats["scales"])
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.splats["quats"])

    @property
    def get_xyz(self):
        return self.splats["means"]

    @property
    def get_opacity(self):
        return torch.sigmoid(self.splats["opacities"])

    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    @property
    def get_covmat(self):
        w, x, y, z = self.get_rotation.unbind(-1)
        scale = self.get_scaling
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = torch.stack([
            torch.stack([1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)], dim=-1),
        ], dim=-2)
        
        e1 = torch.tensor([1,0,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)
        e2 = torch.tensor([0,1,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)

        # Scale local basis
        v1 = e1 * scale[:, [0]]  # (N,3)
        v2 = e2 * scale[:, [1]]  # (N,3)

        # Apply rotation: batch matmul (N,3,3) @ (N,3,1) -> (N,3,1)
        t_u = torch.bmm(R, v1.unsqueeze(-1)).squeeze(-1)  # (N,3)
        t_v = torch.bmm(R, v2.unsqueeze(-1)).squeeze(-1)  # (N,3)

        # Magnitudes
        m_u = torch.linalg.norm(t_u, dim=-1)
        m_v = torch.linalg.norm(t_v, dim=-1)

        # Directions (normalized)
        d_u = t_u / m_u.unsqueeze(-1)
        d_v = t_v / m_v.unsqueeze(-1)

        magnitudes = torch.stack([m_u, m_v], dim=-1)     # (N,2)
        directions = torch.stack([d_u, d_v], dim=1)      # (N,2,3)

        return magnitudes, directions

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.splats["means"].detach().cpu().numpy()
        opacities = self.splats["opacities"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()
        
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        

        means = torch.tensor(xyz, dtype=torch.float, device="cuda")

        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        col_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color")]
        col_names = sorted(col_names, key = lambda x: int(x.split('_')[-1]))
        cols = np.zeros((xyz.shape[0], len(col_names)))
        for idx, attr_name in enumerate(col_names):
            cols[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))
        
        cent = means.mean(0)
        means -= cent.unsqueeze(0)

        self.active_sh_degree = 3
        self.splats = {
            "means":means.cuda().float(),
            "scales":torch.from_numpy(scales).cuda().float(),
            "quats":torch.from_numpy(rots).cuda().float(),
            "opacities":torch.from_numpy(opacities).cuda().float(),
            "sh0":torch.from_numpy(features_dc).cuda().permute(0,2,1).float(),
            "shN":torch.from_numpy(features_extra).cuda().permute(0,2,1).float()
        }
    
    @property
    def point_count(self):
        return self.splats["means"].shape[0]
    

class TraceGaussian(BasicGaussianModel):

    def __init__(self,):
        super().__init__()
        self.base_color = [1., 0., 0.]
    

    def process_draw_loop(self, height, radius):
        
        N = 100
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Evenly spaced angles around the circle: [0, 2π)
        angles = torch.linspace(0, 2 * np.pi, N + 1, device=device)[:-1]
        
        # Positions on the circle at given height
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.full((N,), float(height), device=device)
        means3D = torch.stack([x, y, z], dim=-1)  # (N, 3)
        
        # Identity rotation quaternion (w, x, y, z) = (1, 0, 0, 0)
        rotations = torch.zeros((N, 4), device=device)
        rotations[:, 0] = 1.0
        
        # Opacity = 1
        opacity = torch.ones((N, 1), device=device)
        
        # Red color
        colors = torch.zeros((N, 16, 3), device=device)
        colors[:, 0] = 1.0  # R channel
        
        # Fixed scale
        scales = torch.full((N, 3), 0.01, device=device)
        
        return means3D, rotations, opacity, colors, scales