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

    def __init__(self, color=[1., 0., 0.]):
        super().__init__()
        self.base_color = color
        
        self.height = None
        self.radius = None
        self.N = None
        
        self.rays = None
        
        self.record = None
        self.mesh_data = []

    def generate_loop(self, height, radius, N, raw_return=False):
        # Evenly spaced angles around the circle: [0, 2π)
        angles = torch.linspace(0, 2 * np.pi, N + 1, device="cuda")[:-1]
        
        # Positions on the circle at given height
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.full((N,), float(height), device="cuda")
        means = torch.stack([x, y, z], dim=-1)  # (N, 3)
        
        direction = -means / radius
        direction[:, -1] = 0
        
        
        if raw_return:
            return means,direction
        
        self.rays={
            "means":means,
            "directions":direction
        }
        
    def process_draw_ray_samples(self):        
        # Get globals
        N = self.N
                
        # Arbitrary quaternion 
        rotations = torch.zeros((N, 4), device="cuda")
        rotations[:, 0] = 1.0
        
        # Fixed dense opacity
        opacity = torch.ones((N, 1), device="cuda")
        
        # Arbitrary color for readability
        colors = torch.zeros((N, 16, 3), device="cuda")
        colors[:, 0, 0] = self.base_color[0]  # R channel
        colors[:, 0, 1] = self.base_color[1]  # G channel
        colors[:, 0, 2] = self.base_color[2]  # B channel
        
        # Arbitrary scale (placeholder that is rescaled based on GUI Viewing/Editor settings)
        scales = torch.full((N, 3), 0.01, device="cuda")
        
        return self.rays["means"], rotations, opacity, colors, scales
    
    def process_loop(self, pc, height, radius, N = 100):        
        # Set globals
        self.height = height
        self.radius = radius
        self.N = N
        
        # Compute self.rays["means"] and self.rays["directions"]
        self.generate_loop(height, radius, N)
        
        # Get the visual points of the ray origins
        means, rotations, opacity, colors, scales = self.process_draw_ray_samples()

        # Compute the of the samples with intersection with the pc
        means_int, rotations_int, opacity_int, colors_int, scales_int = self.process_ray_intersection(pc)
        
        # Temporarily save for later
        self.record = (means_int, rotations_int, opacity_int, colors_int, scales_int)
        
        mean_cat = [means, means_int]
        rot_cat = [rotations, rotations_int]
        opac_cat = [opacity, opacity_int]
        col_cat = [colors, colors_int]
        sca_cat = [scales, scales_int]

        if self.mesh_data != []:
            for meta in self.mesh_data:
                if meta["type"] == "user" or meta["type"] == "inbetweens":
                    obj = meta["record"]
                    mean_cat.append(obj[0]) 
                    rot_cat.append(obj[1]) 
                    opac_cat.append(obj[2]) 
                    col_cat.append(obj[3]) 
                    sca_cat.append(obj[4]) 
            
        
        means = torch.cat(mean_cat, dim=0)
        rotations = torch.cat(rot_cat, dim=0)
        opacity = torch.cat(opac_cat, dim=0)
        colors = torch.cat(col_cat, dim=0)
        scales = torch.cat(sca_cat, dim=0)

        return means, rotations, opacity, colors, scales


    def process_ray_intersection(self, pc, ray_origins=None, ray_direction=None, height=None):
        if height is None:
            height=self.height
        if ray_origins is None and ray_direction is None:
            ray_origins = self.rays["means"]
            ray_direction = self.rays["directions"]
            
        # Initial curll based on expected point height
        pc_means = pc.get_xyz
        diff = (pc_means[:, -1].max() - pc_means[:, -1].min())*0.1
        inbounds = (pc_means[:, -1] - height).abs() < diff 
        
        means = pc_means[inbounds]
        inv_cov = pc.inv_covariance[inbounds]
        
        
        N = ray_origins.shape[0]
        M = means.shape[0]
        
        o = ray_origins.unsqueeze(1)    # (N, 1, 3)
        d = ray_direction.unsqueeze(1)       # (N, 1, 3)
        mu = means.unsqueeze(0)  # (1, M, 3)
        delta = o - mu                  # (N, M, 3)

        d_exp = d.expand(N, M, 3)          # (N, M, 3) — use this everywhere
        
        # --- Compute quadratic coefficients A, B, C ---
        # A = d^T Σ⁻¹ d  →  (N, M)
        # Using einsum: for each (n,m), contract d[n] with inv_cov[m] with d[n]
        inv_cov = inv_cov.unsqueeze(0)  # (1, M, 3, 3)

        # Σ⁻¹ d → (N, M, 3)
        inv_cov_d     = torch.einsum('nmij,nmj->nmi', inv_cov.expand(N, -1, -1, -1), d_exp)
        inv_cov_delta = torch.einsum('nmij,nmj->nmi', inv_cov.expand(N, -1, -1, -1), delta)
        A = (d_exp * inv_cov_d).sum(-1)
        B = 2 * (delta * inv_cov_d).sum(-1)
        C = (delta * inv_cov_delta).sum(-1)

        # --- Analytic peak: t* = -B / 2A ---
        t_star = -B / (2 * A + 1e-8)                        # (N, M)

        # Only consider hits in front of the ray origin
        valid = t_star > 0                                   # (N, M)

        # --- Mahalanobis distance at peak ---
        d2_min = C - (B ** 2) / (4 * A + 1e-8)             # (N, M)

        # Gaussian weight at peak
        weight = torch.exp(-0.5 * d2_min)                   # (N, M)

        # Threshold: only keep strong hits
        WEIGHT_THRESH = 0.2
        hits = valid & (weight > WEIGHT_THRESH)              # (N, M)

        # --- Per ray: find the closest (smallest t*) hit ---
        t_star_masked = t_star.clone()
        t_star_masked[~hits] = float('inf')
        best_t, best_m = t_star_masked.min(dim=1)           # (N,), (N,)

        hit_rays = best_t < float('inf')                    # (N,) — rays that hit anything

        # --- Compute intersection points ---
        # p* = o + t* d
        int_points = ray_origins + best_t.unsqueeze(1) * ray_direction   # (N, 3)

        # --- Build outputs only for hit rays ---
        int_points = int_points[hit_rays]                   # (N', 3)
        N_hit = int_points.shape[0]

        rotations_int = torch.zeros((N_hit, 4), device="cuda")
        rotations_int[:, 0] = 1.0
        opacity_int = torch.ones((N_hit, 1), device="cuda")
        colors_int = torch.zeros((N_hit, 16, 3), device="cuda")
        colors_int[:, 0, 1] = 1.0  # Highlight intersections in red
        scales_int = torch.full((N_hit, 3), 0.01, device="cuda")

        return int_points, rotations_int, opacity_int, colors_int, scales_int
    
    
    def set_loop(self, pc):
        content={
            "type":"user",
            "height": self.height,
            "radius":self.radius,
            "N":self.N,
            "record":self.record
        }
        self.mesh_data.append(content)
        
        targets = []
        for meta in self.mesh_data:
            if meta["type"] == "user": 
                targets.append(meta)
                
        if len(targets) >= 2:
            # subsample N heights between the last two
            A = targets[-2]
            B = targets[-1]
            
            sample_heights = B["height"] - A["height"]
            sample_heights = [A["height"] + sample_heights*(i/10) for i in range(1, 10)]
            sample_radius = [A["radius"]*(i/10) +B["radius"]*((10-i)/10) for i in range(1, 10)]
            
            for sh, sr in zip(sample_heights, sample_radius):
                ray_o, ray_d = self.generate_loop(sh, sr, A["N"], raw_return=True)
                means_int, rotations_int, opacity_int, colors_int, scales_int = self.process_ray_intersection(pc, ray_origins=ray_o, ray_direction=ray_d, height=sh)
                
                colors_int = colors_int*0
                colors_int[:, 0, 0] = 1.
                content={
                    "type":"inbetweens",
                    "height":sh,
                    "radius":sr,
                    "N":A["N"],
                    "record":(means_int, rotations_int, opacity_int, colors_int, scales_int)
                }
                self.mesh_data.append(content)
                
            self.export_blender_obj(
                "./settled.obj",
                blender_exe="blender",          # or full path e.g. "/usr/bin/blender"
                drop_height=0.5,
                sim_frames=250,
            )
                
    def export_blender_obj(
        self,
        out_path,
        blender_exe="blender",
        drop_height=0.5,
        sim_frames=250,
        workdir=None,
    ):
        """
        Simulate the cut cylindrical mesh dropping onto a ground plane using
        Blender's cloth sim, and export the settled mesh as an OBJ to out_path.
        """
        import subprocess, tempfile

        # --- Gather mesh grid ---
        loops = [m for m in self.mesh_data if m["type"] in ("user", "inbetweens")]
        if len(loops) < 2:
            print("Need at least 2 loops.")
            return
        loops = sorted(loops, key=lambda m: m["height"])
        N = loops[0]["N"]; H = len(loops)
        P = np.stack(
            [lp["record"][0].detach().cpu().numpy() for lp in loops], axis=0
        ).astype(np.float64)
        P[..., :2] -= P[..., :2].reshape(-1, 2).mean(axis=0)
        P[..., 2] -= P[..., 2].min()
        P[..., 2] += drop_height

        # --- Workdir and input OBJ ---
        workdir = workdir or tempfile.mkdtemp(prefix="cloth_export_")
        os.makedirs(workdir, exist_ok=True)
        in_obj = os.path.join(workdir, "mesh_in.obj")
        script = os.path.join(workdir, "run.py")

        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with open(in_obj, "w") as f:
            for i in range(H):
                for k in range(N):
                    x, y, z = P[i, k]
                    f.write(f"v {x} {y} {z}\n")
            def vid(i, k): return i * N + k + 1
            for i in range(H - 1):
                for k in range(N - 1):
                    f.write(f"f {vid(i,k)} {vid(i,k+1)} {vid(i+1,k+1)} {vid(i+1,k)}\n")

        # --- Blender script. IMPORTANT: every line inside the f-string must
        # start at column 0 so we don't write indented Python to disk. ---
        blender_script = f'''\
import bpy, sys, os, traceback

IN_OBJ     = r"{in_obj}"
OUT_OBJ    = r"{out_path}"
SIM_FRAMES = {sim_frames}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = SIM_FRAMES

    # Ground
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.active_object
    bpy.ops.object.modifier_add(type='COLLISION')
    for attr, val in [("damping", 0.5), ("friction_factor", 0.8),
                      ("thickness_outer", 0.01), ("thickness_inner", 0.01)]:
        if hasattr(ground.collision, attr):
            setattr(ground.collision, attr, val)

    # Import cloth
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=IN_OBJ)
    else:
        bpy.ops.import_scene.obj(filepath=IN_OBJ)
    cloth = bpy.context.selected_objects[0]
    cloth.name = "Cloth"
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.shade_smooth()
    bpy.ops.object.modifier_add(type='CLOTH')
    cs = cloth.modifiers["Cloth"].settings
    col = cloth.modifiers["Cloth"].collision_settings
    cs.mass = 0.5
    cs.tension_stiffness = cs.compression_stiffness = cs.shear_stiffness = 40.0
    cs.bending_stiffness = 2.0
    cs.tension_damping = cs.compression_damping = cs.shear_damping = 5.0
    cs.bending_damping = 2.0
    cs.air_damping = 1.5
    col.use_collision = True
    col.use_self_collision = True
    col.self_distance_min = 0.002
    col.distance_min = 0.003
    col.collision_quality = 4

    # Drive the sim
    print("Simulating", SIM_FRAMES, "frames...")
    for frm in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frm)
        if frm % 25 == 0:
            print("  frame", frm)

    # Capture simulated mesh at final frame and bake it into the object
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = cloth.evaluated_get(depsgraph)
    mesh_eval = bpy.data.meshes.new_from_object(eval_obj)
    cloth.modifiers.clear()
    cloth.data = mesh_eval

    # Select just the cloth for export
    bpy.ops.object.select_all(action='DESELECT')
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth

    # --- Try Blender's OBJ exporters, then fall back to a manual writer ---
    exported = False

    if hasattr(bpy.ops.wm, "obj_export"):
        try:
            res = bpy.ops.wm.obj_export(
                filepath=OUT_OBJ,
                export_selected_objects=True,
                export_materials=False,
                apply_modifiers=True,
            )
            print("bpy.ops.wm.obj_export ->", res)
            exported = os.path.exists(OUT_OBJ)
        except Exception as e:
            print("bpy.ops.wm.obj_export raised:", e)

    if not exported and hasattr(bpy.ops, "export_scene") and hasattr(bpy.ops.export_scene, "obj"):
        try:
            res = bpy.ops.export_scene.obj(
                filepath=OUT_OBJ,
                use_selection=True,
                use_materials=False,
            )
            print("bpy.ops.export_scene.obj ->", res)
            exported = os.path.exists(OUT_OBJ)
        except Exception as e:
            print("bpy.ops.export_scene.obj raised:", e)

    if not exported:
        # Manual writer — no dependency on Blender's IO addons
        print("Falling back to manual OBJ writer")
        mesh = cloth.data
        world = cloth.matrix_world
        with open(OUT_OBJ, "w") as f:
            f.write("# manual export from Blender cloth sim\\n")
            for v in mesh.vertices:
                co = world @ v.co
                f.write("v %.6f %.6f %.6f\\n" % (co.x, co.y, co.z))
            for poly in mesh.polygons:
                idx = " ".join(str(i + 1) for i in poly.vertices)
                f.write("f " + idx + "\\n")
        exported = os.path.exists(OUT_OBJ)

    if not exported:
        print("ERROR: failed to write", OUT_OBJ)
        sys.exit(1)

    print("Wrote", OUT_OBJ)

except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

        with open(script, "w") as fh:
            fh.write(blender_script)

        cmd = [blender_exe, "--background", "--python", script]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("===== Blender stdout =====")
        print(result.stdout)
        print("===== Blender stderr =====")
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Blender exited with code {result.returncode}")
        if not os.path.exists(out_path):
            raise RuntimeError(f"Blender ran but did not produce {out_path}")

        print(f"Exported settled mesh to {out_path}")
        return out_path