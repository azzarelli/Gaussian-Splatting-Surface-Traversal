import torch
import numpy as np
import os

from scene.base_gaussian import BasicGaussianModel

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
        
        self.stress_map = None

    def generate_loop(self, height, radius, N, raw_return=False, pc=None):
        # Compute local center (cx, cy) from points near this height
        if pc is not None:
            pc_means = pc.get_xyz
            diff = (pc_means[:, -1].max() - pc_means[:, -1].min()) * 0.1
            inbounds = (pc_means[:, -1] - height).abs() < diff
            local = pc_means[inbounds]
            if local.shape[0] > 0:
                cx = local[:, 0].mean()
                cy = local[:, 1].mean()
            else:
                cx = torch.tensor(0.0, device="cuda")
                cy = torch.tensor(0.0, device="cuda")
        else:
            cx = torch.tensor(0.0, device="cuda")
            cy = torch.tensor(0.0, device="cuda")

        # Evenly spaced angles around the circle: [0, 2π)
        angles = torch.linspace(0, 2 * np.pi, N + 1, device="cuda")[:-1]

        # Positions on the circle at given height, centered on (cx, cy)
        x = cx + radius * torch.cos(angles)
        y = cy + radius * torch.sin(angles)
        z = torch.full((N,), float(height), device="cuda")
        means = torch.stack([x, y, z], dim=-1)  # (N, 3)

        # Direction = pointing from each ring point toward the local center (cx, cy)
        direction = torch.stack([cx - x, cy - y, torch.zeros(N, device="cuda")], dim=-1)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        if raw_return:
            return means, direction

        self.rays = {
            "means": means,
            "directions": direction,
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
        content = {
            "type": "user",
            "height": self.height,
            "radius": self.radius,
            "N": self.N,
            "record": self.record
        }
        self.mesh_data.append(content)

        targets = [m for m in self.mesh_data if m["type"] == "user"]
        if len(targets) < 2:
            return

        A = targets[-2]
        B = targets[-1]

        N_loops = 30
        sample_heights = B["height"] - A["height"]
        sample_heights = [A["height"] + sample_heights * (i / N_loops) for i in range(1, N_loops)]
        sample_radius  = [A["radius"] * (i / N_loops) + B["radius"] * ((N_loops - i) / N_loops) for i in range(1, N_loops)]

        for sh, sr in zip(sample_heights, sample_radius):
            ray_o, ray_d = self.generate_loop(sh, sr, A["N"], raw_return=True, pc=pc)
            means_int, rotations_int, opacity_int, colors_int, scales_int = self.process_ray_intersection(
                pc, ray_origins=ray_o, ray_direction=ray_d, height=sh
            )
            colors_int = colors_int * 0
            colors_int[:, 0, 0] = 1.
            self.mesh_data.append({
                "type": "inbetweens",
                "height": sh, "radius": sr, "N": A["N"],
                "record": (means_int, rotations_int, opacity_int, colors_int, scales_int)
            })

        rings = [A["record"][0].unsqueeze(0)]
        for meta in self.mesh_data:
            if meta["type"] == "inbetweens":
                rings.append(meta["record"][0].unsqueeze(0))
        rings.append(B["record"][0].unsqueeze(0))
        mesh = torch.cat(rings, dim=0)   # (M, N, 3) on cuda

        device = mesh.device
        M_rings, N_samp, _ = mesh.shape
        V = M_rings * N_samp
        verts_3d = mesh.reshape(-1, 3)   # (V, 3) on cuda

        def vid(r, i):
            return r * N_samp + i

        # ---- Build triangle index tensor (on GPU) ----
        tris = []
        for r in range(M_rings - 1):
            for i in range(N_samp - 1):
                tris.append([vid(r, i),     vid(r, i + 1),     vid(r + 1, i + 1)])
                tris.append([vid(r, i),     vid(r + 1, i + 1), vid(r + 1, i)])
        triangles = torch.tensor(tris, dtype=torch.long, device=device)   # (T, 3)
        T = triangles.shape[0]

        # ---- Isometric 2D template per triangle (vectorised on GPU) ----
        p = verts_3d[triangles]                              # (T, 3, 3)
        p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]
        v01 = p1 - p0
        v02 = p2 - p0
        L01 = v01.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        e1 = v01 / L01
        x_t = (v02 * e1).sum(-1, keepdim=True)
        y_sq = (v02 * v02).sum(-1, keepdim=True) - x_t * x_t
        y_t = y_sq.clamp(min=0.0).sqrt()
        ideal_2d = torch.zeros((T, 3, 2), device=device)
        ideal_2d[:, 1, 0] = L01.squeeze(-1)
        ideal_2d[:, 2, 0] = x_t.squeeze(-1)
        ideal_2d[:, 2, 1] = y_t.squeeze(-1)

        edge_pairs = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        ideal_edges = ideal_2d[:, edge_pairs[:, 1]] - ideal_2d[:, edge_pairs[:, 0]]   # (T, 3, 2)

        # ---- Initial guess ----
        avg_h = (verts_3d[triangles[:, 1]] - verts_3d[triangles[:, 0]]).norm(dim=-1).mean()
        avg_v = (verts_3d[triangles[:, 2]] - verts_3d[triangles[:, 0]]).norm(dim=-1).mean()
        U = torch.zeros((V, 2), device=device)
        rr = torch.arange(M_rings, device=device).view(M_rings, 1).expand(M_rings, N_samp)
        ii = torch.arange(N_samp, device=device).view(1, N_samp).expand(M_rings, N_samp)
        U[:, 0] = (ii.float() * avg_h).reshape(-1)
        U[:, 1] = -(rr.float() * avg_v).reshape(-1)

        # ---- Build sparse Laplacian on GPU ----
        a_idx = triangles[:, edge_pairs[:, 0]].reshape(-1)
        b_idx = triangles[:, edge_pairs[:, 1]].reshape(-1)
        rows = torch.cat([a_idx, b_idx, a_idx, b_idx])
        cols = torch.cat([a_idx, b_idx, b_idx, a_idx])
        vals = torch.cat([torch.ones_like(a_idx, dtype=torch.float32),
                        torch.ones_like(b_idx, dtype=torch.float32),
                        -torch.ones_like(a_idx, dtype=torch.float32),
                        -torch.ones_like(b_idx, dtype=torch.float32)])

        seam_ids = torch.tensor([vid(r, 0) for r in range(M_rings)], device=device)
        y_pin = seam_ids[0].item()
        y_pin_value = U[y_pin, 1].item()

        def build_L(pin_ids):
            # zero out rows for pinned vertices, then add identity
            keep = ~torch.isin(rows, pin_ids)
            r2 = torch.cat([rows[keep], pin_ids])
            c2 = torch.cat([cols[keep], pin_ids])
            v2 = torch.cat([vals[keep], torch.ones(len(pin_ids), device=device)])
            L = torch.sparse_coo_tensor(torch.stack([r2, c2]), v2, (V, V)).coalesce()
            return L.to_dense()   # dense for solve; V is small (~M*N)

        L_x = build_L(seam_ids)
        L_y = build_L(torch.tensor([y_pin], device=device))

        # ---- ARAP iterations on GPU ----
        n_iters = 30
        for _ in range(n_iters):
            # Local step: best rotation per triangle (batched SVD)
            cur = U[triangles[:, edge_pairs[:, 1]]] - U[triangles[:, edge_pairs[:, 0]]]   # (T, 3, 2)
            H = ideal_edges.transpose(1, 2) @ cur                                          # (T, 2, 2)
            Um_, _, Vt_ = torch.linalg.svd(H)
            Rt = Vt_.transpose(-1, -2) @ Um_.transpose(-1, -2)
            # fix reflections
            det = torch.linalg.det(Rt)
            flip = (det < 0).float().view(-1, 1, 1)
            Vt_fixed = Vt_.clone()
            Vt_fixed[:, -1, :] = Vt_fixed[:, -1, :] * (1 - 2 * flip.squeeze(-1))
            Rt = Vt_fixed.transpose(-1, -2) @ Um_.transpose(-1, -2)

            # Global step: build RHS
            rotated = (Rt.unsqueeze(1) @ ideal_edges.unsqueeze(-1)).squeeze(-1)   # (T, 3, 2)
            rhs = torch.zeros((V, 2), device=device)
            a = triangles[:, edge_pairs[:, 0]]
            b = triangles[:, edge_pairs[:, 1]]
            rhs.index_add_(0, a.reshape(-1), -rotated.reshape(-1, 2))
            rhs.index_add_(0, b.reshape(-1),  rotated.reshape(-1, 2))

            # Solve x with seam pinned to 0
            rhs_x = rhs[:, 0].clone()
            rhs_x[seam_ids] = 0.0
            U[:, 0] = torch.linalg.solve(L_x, rhs_x)

            # Solve y with single anchor
            rhs_y = rhs[:, 1].clone()
            rhs_y[y_pin] = y_pin_value
            U[:, 1] = torch.linalg.solve(L_y, rhs_y)

        # ---- Compute per-triangle stress (area ratio: 2D area / 3D area) ----
        cur = U[triangles[:, edge_pairs[:, 1]]] - U[triangles[:, edge_pairs[:, 0]]]
        e_2d_a = cur[:, 0]   # (T, 2)
        e_2d_b = -cur[:, 2]  # edge from p0 to p2 in 2D
        area_2d = 0.5 * (e_2d_a[:, 0] * e_2d_b[:, 1] - e_2d_a[:, 1] * e_2d_b[:, 0]).abs()
        e_3d_a = verts_3d[triangles[:, 1]] - verts_3d[triangles[:, 0]]
        e_3d_b = verts_3d[triangles[:, 2]] - verts_3d[triangles[:, 0]]
        area_3d = 0.5 * torch.linalg.cross(e_3d_a, e_3d_b).norm(dim=-1)
        stress = (area_2d / area_3d.clamp(min=1e-12)).log().abs()   # 0 = no distortion
        stress = stress / stress.max().clamp(min=1e-6)              # normalise to [0, 1]

        # ---- Rasterise into stress_map image ----
        H_img = 512
        W_img = 1024
        # Map U coords to pixel coords with margin
        u_min = U.min(0).values
        u_max = U.max(0).values
        u_range = (u_max - u_min).clamp(min=1e-6)
        margin = 8
        scale = torch.tensor([(W_img - 2 * margin) / u_range[0],
                            (H_img - 2 * margin) / u_range[1]], device=device)
        s = scale.min()   # preserve aspect ratio
        U_px = (U - u_min) * s
        U_px[:, 0] += margin
        U_px[:, 1] = (H_img - margin) - U_px[:, 1]   # flip y for image coords

        # Build canvas and rasterise each triangle
        canvas = torch.zeros((H_img, W_img, 3), device=device)
        tri_px = U_px[triangles]   # (T, 3, 2)

        # Colour from stress: blue (low) -> red (high)
        colors = torch.zeros((T, 3), device=device)
        colors[:, 0] = stress           # R
        colors[:, 2] = 1.0 - stress     # B

        self._rasterise_triangles(canvas, tri_px, colors)
        self.stress_map = canvas
        
        # ---- Recolor gaussian points based on stress ----
        # Convert per-triangle stress -> per-vertex stress (mean over incident triangles)
        vert_stress = torch.zeros(V, device=device)
        vert_count = torch.zeros(V, device=device)
        for k in range(3):
            vert_stress.index_add_(0, triangles[:, k], stress)
            vert_count.index_add_(0, triangles[:, k], torch.ones_like(stress))
        vert_stress = vert_stress / vert_count.clamp(min=1)   # (V,)

        # Same colour map as the heat map: blue (low) -> red (high)
        vert_colors = torch.zeros((V, 3), device=device)
        vert_colors[:, 0] = vert_stress
        vert_colors[:, 2] = 1.0 - vert_stress

        # Reshape back to (M_rings, N_samp, 3) so we can index per ring
        vert_colors = vert_colors.view(M_rings, N_samp, 3)

        # Walk the rings in the same order they were concatenated into `mesh`:
        #   ring 0  = A (the older "user" loop)
        #   rings 1..M_rings-2 = inbetweens
        #   ring M_rings-1 = B (the newest "user" loop)
        ring_idx = 0

        # Update A's colors_int (in mesh_data)
        A_meta = next(m for m in self.mesh_data
                    if m["type"] == "user" and m is targets[-2])
        self._recolor_record(A_meta, vert_colors[ring_idx])
        ring_idx += 1

        # Update inbetweens (in order of insertion)
        for meta in self.mesh_data:
            if meta["type"] == "inbetweens":
                self._recolor_record(meta, vert_colors[ring_idx])
                ring_idx += 1

        # Update B's colors_int
        B_meta = next(m for m in self.mesh_data
                    if m["type"] == "user" and m is targets[-1])
        self._recolor_record(B_meta, vert_colors[ring_idx])


    def _rasterise_triangles(self, canvas, tri_px, colors):
        """Fill each triangle with a flat colour. canvas: (H, W, 3) on GPU."""
        H, W, _ = canvas.shape
        device = canvas.device
        T = tri_px.shape[0]

        for t in range(T):
            v0, v1, v2 = tri_px[t]
            x_min = int(max(0, torch.floor(torch.min(tri_px[t, :, 0])).item()))
            x_max = int(min(W - 1, torch.ceil(torch.max(tri_px[t, :, 0])).item()))
            y_min = int(max(0, torch.floor(torch.min(tri_px[t, :, 1])).item()))
            y_max = int(min(H - 1, torch.ceil(torch.max(tri_px[t, :, 1])).item()))
            if x_max <= x_min or y_max <= y_min:
                continue

            ys = torch.arange(y_min, y_max + 1, device=device).view(-1, 1)
            xs = torch.arange(x_min, x_max + 1, device=device).view(1, -1)
            # barycentric test
            denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
            if denom.abs() < 1e-8:
                continue
            a = ((v1[1] - v2[1]) * (xs - v2[0]) + (v2[0] - v1[0]) * (ys - v2[1])) / denom
            b = ((v2[1] - v0[1]) * (xs - v2[0]) + (v0[0] - v2[0]) * (ys - v2[1])) / denom
            c = 1 - a - b
            inside = (a >= 0) & (b >= 0) & (c >= 0)
            canvas[y_min:y_max + 1, x_min:x_max + 1][inside] = colors[t]
            
    def _recolor_record(self, meta, ring_colors):
        """Replace the SH-DC color of every gaussian in this record with ring_colors.
        ring_colors: (N_samp, 3) on GPU. record's colors_int has shape (N_hit, 16, 3)
        where N_hit may be less than N_samp because some rays missed.
        """
        means_int, rotations_int, opacity_int, colors_int, scales_int = meta["record"]
        N_hit = colors_int.shape[0]
        # ring_colors has one entry per ray; we wrote stress per-vertex assuming
        # all rays hit. If N_hit < N_samp, just take the first N_hit colours —
        # this matches the order points were kept after the hit-mask filter,
        # which preserves ray order.
        new = ring_colors[:N_hit].to(colors_int.device, colors_int.dtype)
        colors_int = colors_int * 0
        colors_int[:, 0, :3] = new   # SH DC term, RGB
        meta["record"] = (means_int, rotations_int, opacity_int, colors_int, scales_int)
        
    
    