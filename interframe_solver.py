import time
import torch
import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional, Dict, Any
import ipdb

from eval_vis_pcd_traj import create_camera_frustum

from third_party_codes.vggt_long_code import robust_weighted_estimate_sim3

def make_pcd(pts_np, color, special_point_idx=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float32), (pts_np.shape[0], 1))
    )
    if special_point_idx is not None:
        colors = np.asarray(pcd.colors)
        colors[special_point_idx] = np.array([0.0, 0.0, 0.0])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_correspondences(src_points: np.ndarray,
                              tgt_points: np.ndarray,
                              src_color=(1, 0, 0),
                              tgt_color=(0, 1, 0),
                              line_color=(0, 0, 1),
                              background: Tuple[np.ndarray, np.ndarray] = None,
                              window_name="correspondences"):
    """
    Visualize correspondences between two point clouds
    Args:
        src_points: (N,3) source point cloud
        tgt_points: (N,3) target point cloud
        src_color:  source point color (default red)
        tgt_color:  target point color (default green)
        line_color: line color (default blue)
        background: (M,6) background point cloud (x,y,z,r,g,b), r,g,b∈[0,1]
    """
    assert src_points.shape == tgt_points.shape, "Point clouds must correspond one-to-one"
    N = src_points.shape[0]
    geoms = []

    if background is not None:
        bg_points = background[0]
        bg_colors = background[1]/255
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(bg_points)
        pcd_bg.colors = o3d.utility.Vector3dVector(bg_colors)
        geoms.append(pcd_bg)

        # Visualize src_points / tgt_points with small spheres
        radius = 0.005  # Can be adjusted larger/smaller
        resolution = 12
        for p in src_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
            sphere.paint_uniform_color(src_color)
            sphere.translate(p)
            geoms.append(sphere)

        for p in tgt_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
            sphere.paint_uniform_color(tgt_color)
            sphere.translate(p)
            geoms.append(sphere)
    else:
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(src_points)
        pcd_src.paint_uniform_color(src_color)
        geoms.append(pcd_src)

        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(tgt_points)
        pcd_tgt.paint_uniform_color(tgt_color)
        geoms.append(pcd_tgt)

    # Lines (src[i] -> tgt[i])
    all_points = np.vstack([src_points, tgt_points])
    lines = [[i, i + N] for i in range(N)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(line_color)
    geoms.append(line_set)

    o3d.visualization.draw_geometries(geoms, window_name=window_name)

# =========================================================
# 3D linear kernel U(r) = r
# =========================================================
def _kernel_linear_r(dists: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    if eps > 0:
        return torch.sqrt(dists * dists + eps)
    return dists


# =========================================================
# Control point selection: random / voxel / octree
# - Uniform return: cp_idx (n_c,)
# - If the number of selected points > max_control_points, randomly truncate to the upper limit
# =========================================================
@torch.no_grad()
def select_control_points(
    src: torch.Tensor,                  # (N,3)
    max_control_points: int,
    method: str = "voxel",             # "voxel"
    vis = True,
    kwargs: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Returns: cp_idx, aux (aux is a placeholder)
    If kwargs["visualize_cells"]=True, immediately visualize with Open3D after selection:
      - Original points (random subsample, size point_size)
      - Control points (blue, size point_size)
      - voxel / octree leaf AABB wireframes (voxel=green; octree=orange)
    """
    if kwargs is None:
        kwargs = {}
    device = src.device
    N = src.shape[0]

    vis_points_sample = int(kwargs.get("vis_points_sample", 100_000))
    vis_boxes_limit   = int(kwargs.get("vis_boxes_limit", 100_000))
    point_size        = float(kwargs.get("point_size", 10))  

    def _o3d_draw(
        points_np,              # Original points (M,3) numpy
        cp_np,                  # Control points (K,3) numpy
        aabb_mins_np=None,      # (L,3) numpy or None
        aabb_maxs_np=None,      # (L,3) numpy or None
        method_flag="voxel",
        *,
        cp_radius=None,         # None -> auto estimate
        cp_color=(0.2, 0.5, 1.0),
        boxes_limit=10_000
    ):
        import numpy as np, open3d as o3d

        geoms = []

        # 1) Original points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.paint_uniform_color([0.75, 0.75, 0.75])
        geoms.append(pcd)

        # 2) voxel/octree wireframe boxes
        if aabb_mins_np is not None and aabb_maxs_np is not None:
            color = [0.2, 0.9, 0.2] if method_flag == "voxel" else [1.0, 0.6, 0.2]
            L = min(len(aabb_mins_np), boxes_limit)
            for i in range(L):
                mins = aabb_mins_np[i]; maxs = aabb_maxs_np[i]
                x0,y0,z0 = mins; x1,y1,z1 = maxs
                pts = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                                [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], dtype=np.float32)
                lines = np.array([[0,1],[1,2],[2,3],[3,0],
                                [4,5],[5,6],[6,7],[7,4],
                                [0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
                ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                                        lines=o3d.utility.Vector2iVector(lines))
                ls.paint_uniform_color(color)
                geoms.append(ls)

        # 3) Control points (small sphere mesh)
        if cp_np is not None and len(cp_np) > 0:
            # Auto radius: based on point cloud scale or voxel_size (external voxel branch will pass cp_radius)
            if cp_radius is None:
                # Rough estimate: use the diagonal of the bounding box of the original points as a scale reference
                mins = points_np.min(axis=0); maxs = points_np.max(axis=0)
                diag = np.linalg.norm(maxs - mins) + 1e-6
                cp_radius = float(diag) * 0.002  # 0.2% of diagonal

            # Pre-create a unit sphere and scale to cp_radius, then copy/translate later
            base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cp_radius)
            base_sphere.compute_vertex_normals()
            base_sphere.paint_uniform_color(list(cp_color))

            for xyz in cp_np:
                sphere = o3d.geometry.TriangleMesh(base_sphere)
                sphere.translate(xyz, relative=False)
                geoms.append(sphere)

        visl = o3d.visualization.Visualizer()
        visl.create_window()
        for g in geoms:
            visl.add_geometry(g)
        opt = visl.get_render_option()
        opt.point_size = float(point_size)
        visl.run()
        visl.destroy_window()


    if method == "random":
        if N <= max_control_points:
            cp_idx = torch.arange(N, device=device)
        else:
            cp_idx = torch.randperm(N, device=device)[:max_control_points]
        if vis:
            import numpy as np
            src_np = src.detach().cpu().float().numpy()
            if src_np.shape[0] > vis_points_sample:
                sel = np.random.permutation(src_np.shape[0])[:vis_points_sample]
                src_np = src_np[sel]
            cp_np = src[cp_idx].detach().cpu().float().numpy()
            _o3d_draw(src_np, cp_np, None, None, "random")
        return cp_idx

    elif method == "voxel":
        voxel_size: float = float(kwargs.get("voxel_size", 0.3))
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")

        # Voxel integer coordinates
        mins = src.min(dim=0).values
        vcoords = torch.floor((src - mins) / voxel_size).to(torch.long)  # (N,3)

        # Unique rows and get inverse for grouping
        uniq_vox, inverse = torch.unique(vcoords, dim=0, return_inverse=True)
        num_vox = uniq_vox.size(0)

        # —— Select "center point" as representative for each voxel —— (key modification)
        # Center of the voxel each point belongs to
        centers = mins + (vcoords.to(src.dtype) + 0.5) * voxel_size    # (N,3)
        dist2 = ((src - centers) ** 2).sum(dim=1)                      # (N,)

        # Sort by inverse and take argmin for each segment
        order = torch.argsort(inverse)         # (N,)
        inv_sorted = inverse[order]            # (N,)
        dist2_sorted = dist2[order]            # (N,)

        # Segment start positions (first position of each new voxel)
        first_mask = torch.ones_like(inv_sorted, dtype=torch.bool, device=device)
        first_mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
        first_pos = torch.nonzero(first_mask, as_tuple=False).flatten()

        # Segment end positions (one before next segment start; last segment to end)
        last_pos = torch.empty_like(first_pos)
        last_pos[:-1] = first_pos[1:] - 1
        last_pos[-1] = inv_sorted.numel() - 1

        rep_indices = []
        for fp, lp in zip(first_pos.tolist(), last_pos.tolist()):
            # The position of the minimum dist2 in this segment (relative to the sorted array)
            seg = dist2_sorted[fp:lp+1]
            rel = torch.argmin(seg)
            abs_idx_in_order = fp + rel.item()
            rep_indices.append(order[abs_idx_in_order].unsqueeze(0))
        cp_idx = torch.cat(rep_indices, dim=0)  # (~n_voxel,)

        threshold = int(kwargs.get("min_threshold", 5))  # Minimum number of points in a voxel to keep
        # Count points in each voxel
        if threshold > 1:
            counts = torch.bincount(inverse, minlength=num_vox)  # (num_vox,)
            selected_vox_ids = inverse[cp_idx]                   # Voxel indices of control points
            valid_mask = counts[selected_vox_ids] >= threshold   # Keep only voxels meeting the threshold
            cp_idx = cp_idx[valid_mask]
            selected_vox_ids = selected_vox_ids[valid_mask]      # For subsequent visualization
        else:
            selected_vox_ids = inverse[cp_idx]

        # # Control total number
        # if cp_idx.numel() > max_control_points:
        #     perm = torch.randperm(cp_idx.numel(), device=device)[:max_control_points]
        #     cp_idx = cp_idx[perm]
        #     # Synchronize trimming of selected_vox_ids to avoid inconsistency between visualization and cp_idx
        #     selected_vox_ids = selected_vox_ids[perm]

        print(len(cp_idx), "control points selected")
        # Visualization: draw selected voxel AABB + control points + original points
        if vis and cp_idx.numel() > 0:
            import numpy as np
            src_np = src.detach().cpu().float().numpy()
            if src_np.shape[0] > vis_points_sample:
                sel = np.random.permutation(src_np.shape[0])[:vis_points_sample]
                src_np = src_np[sel]
            cp_np = src[cp_idx].detach().cpu().float().numpy()

            leaf_vox = uniq_vox[selected_vox_ids]       # (L,3)
            aabb_min = mins + leaf_vox.to(src.dtype) * voxel_size
            aabb_max = aabb_min + voxel_size

            _o3d_draw(
                src_np,
                cp_np,
                aabb_min.detach().cpu().float().numpy(),
                aabb_max.detach().cpu().float().numpy(),
                method_flag="voxel",
            )

        return cp_idx

    else:
        raise ValueError(f"Unknown control_point_method: {method}")




def select_control_points_np(
    src: np.ndarray,                 # (N,3) float
    method: str = "voxel",           # only support "voxel"
    vis: bool = False,
    kwargs: Optional[Dict] = None,
    given_control_points: Optional[np.ndarray] = None,  # (G,3) or None
    keep_all_given: bool = False,  # whether to keep all given points
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxel-based control point selection (NumPy version, supports prioritizing given_control_points)
    Returns:
      kept_given_idx:  (K_g,)  indices of kept points in given_control_points
      new_cp_src_idx:  (K_s,)  indices of newly created control points in src
    Visualization: given points and new points have different colors
    """
    if kwargs is None:
        kwargs = {}
    assert src.ndim == 2 and src.shape[1] == 3, "src must be (N,3)"
    if given_control_points is not None:
        assert given_control_points.ndim == 2 and given_control_points.shape[1] == 3, \
            "given_control_points must be (G,3)"



    vis_points_sample = int(kwargs.get("vis_points_sample", 100_000))
    vis_boxes_limit   = int(kwargs.get("vis_boxes_limit", 100_000))
    point_size        = float(kwargs.get("point_size", 10.0))
    voxel_size        = float(kwargs.get("voxel_size", 0.3))
    min_threshold     = int(kwargs.get("min_threshold", 1))  # minimum number of points per voxel (for src points)





    if method != "voxel":
        raise ValueError("This modified version currently supports method='voxel' only.")

    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    N = src.shape[0]
    mins = src.min(axis=0)
    vcoords = np.floor((src - mins) / voxel_size).astype(np.int64)  # (N,3)

    # Unique voxels and grouping
    uniq_vox, inverse = np.unique(vcoords, axis=0, return_inverse=True)  # inverse: (N,)
    num_vox = uniq_vox.shape[0]

    # Filter voxels by point count (only for src)
    counts = np.bincount(inverse, minlength=num_vox)
    valid_vox_mask = counts >= max(1, min_threshold)
    valid_vox_ids = np.flatnonzero(valid_vox_mask)  # Keep only these voxels

    # Voxel centers
    centers_all = mins + (uniq_vox.astype(src.dtype) + 0.5) * voxel_size  # (num_vox,3)
    # Squared distance of each src point to its voxel center
    centers_src = centers_all[inverse]                                    # (N,3)
    dist2 = np.sum((src - centers_src) ** 2, axis=1)                      # (N,)

    # For each valid voxel, select the "closest to center" src representative (for voxels not hit by given points)
    # Approach: sort by inverse, segment-wise argmin
    order = np.argsort(inverse)              # (N,)
    inv_sorted = inverse[order]              # (N,)
    dist2_sorted = dist2[order]              # (N,)

    first_mask = np.ones_like(inv_sorted, dtype=bool)
    first_mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first_pos = np.flatnonzero(first_mask)   # segment start (first item of each voxel)
    last_pos = np.empty_like(first_pos)
    if len(first_pos) > 1:
        last_pos[:-1] = first_pos[1:] - 1
    last_pos[-1] = inv_sorted.size - 1

    # Compute representative src index for each voxel (all voxels)
    rep_src_idx_all = -np.ones(num_vox, dtype=np.int64)
    for fp, lp in zip(first_pos, last_pos):
        vox_id = inv_sorted[fp]
        seg = dist2_sorted[fp:lp+1]
        rel = np.argmin(seg)
        abs_idx_in_order = fp + int(rel)
        rep_src_idx_all[vox_id] = order[abs_idx_in_order]
    # Only care about valid voxels
    rep_src_idx_valid = rep_src_idx_all[valid_vox_ids]

    # ============= Handle given_control_points hitting voxels =============
    kept_given_idx_list = []
    new_cp_src_idx_list = []

    if given_control_points is not None and given_control_points.shape[0] > 0:
        # Compute voxel indices for given points (using src mins/voxel_size)
        gv = np.floor((given_control_points - mins) / voxel_size).astype(np.int64)  # (G,3)

        # Build a mapping from voxel to list of given point indices
        from collections import defaultdict
        vox_to_given = defaultdict(list)
        for gi, vc in enumerate(gv):
            # Find if this vc is in the set of valid voxels
            # Use (uniq_vox == vc).all(axis=1) to find index; for efficiency use a dictionary
            # First build a hash table: voxel triplet -> voxel row index
            # Build once
            # (To avoid rebuilding in the loop, we build it in advance)
            pass
        # Build voxel hash table
        vox_hash = {tuple(v): i for i, v in enumerate(uniq_vox)}
        valid_vox_set = set(valid_vox_ids.tolist())

        for gi, vc in enumerate(gv):
            key = tuple(vc.tolist())
            if key in vox_hash:
                vox_id = vox_hash[key]
                if vox_id in valid_vox_set:
                    vox_to_given[vox_id].append(gi)

        # For each valid voxel: if there are given points hitting it -> keep all these given points; otherwise use src representative point
        for vox_id in valid_vox_ids:
            if vox_id in vox_to_given and not keep_all_given:
                kept_given_idx_list.extend(vox_to_given[vox_id])  # Keep all
            else:
                # Create a new control point: use the representative src point of this voxel
                rep_idx = rep_src_idx_all[vox_id]
                if rep_idx >= 0:
                    new_cp_src_idx_list.append(rep_idx)
    else:
        # No given points: use src representative points for all valid voxels
        for vox_id in valid_vox_ids:
            rep_idx = rep_src_idx_all[vox_id]
            if rep_idx >= 0:
                new_cp_src_idx_list.append(rep_idx)

    if not keep_all_given:
        kept_given_idx = np.array(sorted(set(kept_given_idx_list)), dtype=np.int64)
    else:
        if given_control_points is not None:
            kept_given_idx = np.array([i for i in range(given_control_points.shape[0])], dtype=np.int64)
        else:
            kept_given_idx = np.array([], dtype=np.int64)
        

    new_cp_src_idx = np.array(sorted(set(new_cp_src_idx_list)), dtype=np.int64)

    if vis:
        try:
            import open3d as o3d
            # Sample original points for display
            src_vis = src
            if src_vis.shape[0] > vis_points_sample:
                sel = np.random.permutation(src_vis.shape[0])[:vis_points_sample]
                src_vis = src_vis[sel]

            geoms = []

            # Original points (gray)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(src_vis)
            pcd.paint_uniform_color([0.75, 0.75, 0.75])
            geoms.append(pcd)

            # Draw valid voxel AABB wireframes (green)
            aabb_min = mins + uniq_vox[valid_vox_ids].astype(src.dtype) * voxel_size
            aabb_max = aabb_min + voxel_size
            L = min(aabb_min.shape[0], vis_boxes_limit)
            for i in range(L):
                mn = aabb_min[i]; mx = aabb_max[i]
                x0,y0,z0 = mn; x1,y1,z1 = mx
                pts = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                                [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], dtype=np.float32)
                lines = np.array([[0,1],[1,2],[2,3],[3,0],
                                  [4,5],[5,6],[6,7],[7,4],
                                  [0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
                ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                                          lines=o3d.utility.Vector2iVector(lines))
                ls.paint_uniform_color([0.2, 0.9, 0.2])
                geoms.append(ls)

            # Given control points (blue)
            if kept_given_idx.size > 0 and given_control_points is not None:
                if not keep_all_given:
                    cp_given = given_control_points[kept_given_idx]
                else:
                    cp_given = given_control_points
                spheres = _spheres_from_points(cp_given, src_vis, color=(0.2, 0.5, 1.0))
                geoms.extend(spheres)

            # New control points (orange)
            if new_cp_src_idx.size > 0:
                cp_new = src[new_cp_src_idx]
                spheres = _spheres_from_points(cp_new, src_vis, color=(1.0, 0.6, 0.2))
                geoms.extend(spheres)

            visl = o3d.visualization.Visualizer()
            visl.create_window(f"{cp_given.size} given + {new_cp_src_idx.size} new control points (voxel)")
            for g in geoms:
                visl.add_geometry(g)
            opt = visl.get_render_option()
            opt.point_size = float(point_size)
            visl.run()
            visl.destroy_window()

        except ImportError:
            print("[select_control_points_np] open3d not installed; skip visualization.")

    return kept_given_idx, new_cp_src_idx


# —— vis helper ——
def _spheres_from_points(points_np: np.ndarray,
                         cloud_for_scale: np.ndarray,
                         color=(0.2, 0.5, 1.0)):
    import open3d as o3d
    if points_np.size == 0:
        return []
    mins = cloud_for_scale.min(axis=0)
    maxs = cloud_for_scale.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins) + 1e-6)
    radius = diag * 0.002  # 0.2% of the diagonal
    base = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    base.compute_vertex_normals()
    base.paint_uniform_color(list(color))
    geoms = []
    for p in points_np:
        s = o3d.geometry.TriangleMesh(base)
        s.translate(p, relative=False)
        geoms.append(s)
    return geoms



def _bilinear_depth_sample(depth: np.ndarray, u: float, v: float, conf_mask: np.ndarray=None) -> float:
    """
    Bilinear sample depth[v,u]. If conf_mask is not None, only use neighbors with mask True for interpolation.
    - depth: (H, W), depth (must be positive and finite)
    - u: column coordinate (x), v: row coordinate (y), can be fractional
    - conf_mask: (H, W) boolean array; if provided, only take True neighbors and normalize weights
    Returns: scalar depth; if out of bounds/no valid neighbors/invalid depth, returns np.nan
    """
    H, W = depth.shape
    # Need four neighbors, so require to be within [0, W-1)×[0, H-1)
    if not (0.0 <= u < W-1 and 0.0 <= v < H-1):
        return np.nan

    u0 = int(np.floor(u)); v0 = int(np.floor(v))
    u1 = u0 + 1;          v1 = v0 + 1
    du = u - u0;          dv = v - v0

    # four neighbors and basic bilinear weights
    z00 = depth[v0, u0]; w00 = (1 - du) * (1 - dv)
    z10 = depth[v0, u1]; w10 = du       * (1 - dv)
    z01 = depth[v1, u0]; w01 = (1 - du) * dv
    z11 = depth[v1, u1]; w11 = du       * dv

    zs   = np.array([z00, z10, z01, z11], dtype=float)
    ws   = np.array([w00, w10, w01, w11], dtype=float)

    # Basic validity: depth must be finite and >0
    valid = np.isfinite(zs) & (zs > 0)

    # Further filter by confidence mask (if provided)
    if conf_mask is not None:
        if conf_mask.shape != depth.shape or conf_mask.dtype != np.bool_:
            raise ValueError("conf_mask must be a boolean array with the same shape as depth")
        mask_vals = np.array([conf_mask[v0, u0], conf_mask[v0, u1],
                              conf_mask[v1, u0], conf_mask[v1, u1]], dtype=bool)
        valid &= mask_vals

    if not np.any(valid):
        return np.nan

    # Only use valid neighbors, normalize weights
    zs_valid = zs[valid]
    ws_valid = ws[valid]
    wsum = ws_valid.sum()
    if wsum <= 0:
        return np.nan

    z = float(np.dot(zs_valid, ws_valid) / wsum)
    return z




from typing import List, Optional, Tuple
import numpy as np

def project(
    points_world: np.ndarray,           # (N,3)
    
    cam2world: np.ndarray,              # (C,4,4)
    intrinsics: np.ndarray,             # (C,3,3)
    points_world_check: np.ndarray,     # (C,H,W,3)
    H: int, W: int,
    valid_mask: np.ndarray=None,        # (C,H,W)
    max_error: float = 0.01,            # maximum allowed error ratio (relative to scene scale)
    neighbor_size: int = 3,             # neighborhood radius (pixels)
) -> List[Optional[Tuple[int, float, float]]]:
    """
    For each 3D point Pw, among all cameras:
      1) World->Camera to get (Xc,Yc,Zc), skip if Zc <= 0
      2) Project to pixel (u,v)
      3) Search in square neighborhood [(u±neighbor_size),(v±neighbor_size)]
         for the point in points_world_check[cid,vi,ui] with minimum 3D distance to Pw
      4) If minimum distance < max_error, record (cid, ui, vi)
    Return the best camera and pixel coordinates for each point; None if no valid camera.
    """
    assert points_world.ndim == 2 and points_world.shape[1] == 3
    assert cam2world.ndim == 3 and cam2world.shape[1:] == (4, 4)
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3)
    assert points_world_check.ndim == 4 and points_world_check.shape[3] == 3
    assert neighbor_size >= 0
    

    C = cam2world.shape[0]
    N = points_world.shape[0]

    Pw_h = np.concatenate([points_world, np.ones((N, 1), dtype=points_world.dtype)], axis=1)
    world2cam = np.linalg.inv(cam2world)

    out: List[Optional[Tuple[int, float, float]]] = [None] * N

    for i in range(N):
        pwh = Pw_h[i]
        Pw = points_world[i]
        best: Optional[Tuple[int, float, float]] = None
        best_err = np.inf

        for cid in range(C):
            W2C = world2cam[cid]
            K   = intrinsics[cid]

            # World -> Camera coordinates
            Xc, Yc, Zc = (W2C @ pwh)[:3]
            if Zc <= 0.0:
                # print("Zc <= 0")
                continue  # Behind the camera
            # Project to pixel
            xz = Xc / Zc
            yz = Yc / Zc
            uv1 = K @ np.array([xz, yz, 1.0], dtype=float)
            u, v = float(uv1[0]), float(uv1[1])

            # Coarse boundary check (considering neighborhood)
            if u < -neighbor_size or v < -neighbor_size or u >= W + neighbor_size or v >= H + neighbor_size:
                # print("Out of bounds")
                continue

            u0, v0 = int(np.floor(u)), int(np.floor(v))

            # Neighborhood range
            u_min, u_max = max(0, u0 - neighbor_size), min(W - 1, u0 + neighbor_size)
            v_min, v_max = max(0, v0 - neighbor_size), min(H - 1, v0 + neighbor_size)

            for vi in range(v_min, v_max + 1):
                for ui in range(u_min, u_max + 1):
                    if valid_mask is not None and not valid_mask[cid, vi, ui]:
                        # print("Invalid pixel")
                        continue

                    Pw_chk = points_world_check[cid, vi, ui]
                    if not np.isfinite(Pw_chk).all():
                        # print("Invalid 3D point")
                        continue

                    err = float(np.linalg.norm(Pw_chk - Pw))
                    if err < max_error and err < best_err:
                        best_err = err
                        best = (cid, ui, vi)
        out[i] = best

    return out

def unproject(
    pixels_list: List[Optional[Tuple[int, float, float]]],  # [(cid,u,v) | None, ...]；u,v 可为小数
    cam2world: np.ndarray,                                  # (C,4,4)
    intrinsics: np.ndarray,                                 # (C,3,3)
    depths: np.ndarray,                                      # (C,H,W)
    use_subpixel: bool = False
) -> np.ndarray:
    """
    Use bilinear interpolated depth to unproject a list of pixels to world coordinates.
    Returns (M,3) only skipping None (assumes no out-of-bounds / invalid depth).
    """
    assert cam2world.ndim == 3 and cam2world.shape[1:] == (4,4)
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3,3)
    assert depths.ndim == 3
    C, H, W = depths.shape
    
    if isinstance(pixels_list, np.ndarray) and pixels_list.dtype == int:
        use_subpixel = False

    pts_world = []

    for item in pixels_list:
        if item is None:
            continue
        cid, u, v = item

        K   = intrinsics[cid]      # (3,3)
        C2W = cam2world[cid]       # (4,4)
        Rcw = C2W[:3, :3]
        tcw = C2W[:3, 3]

        if use_subpixel:
            # Bilinear depth sampling
            z = _bilinear_depth_sample(depths[cid], float(u), float(v))
        else:
            # Nearest neighbor depth sampling
            ui = int(np.rint(u)); vi = int(np.rint(v))
            z = depths[cid, vi, ui]

        # Unproject to camera coordinates
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        x_n = (float(u) - cx) / fx
        y_n = (float(v) - cy) / fy
        Pc  = np.array([x_n * z, y_n * z, z], dtype=float)  # (3,)

        # Camera -> World
        Pw = Rcw @ Pc + tcw
        pts_world.append(Pw)

    if len(pts_world) == 0:
        return np.zeros((0,3), dtype=float)
    return np.vstack(pts_world)

class TPS3DModel:
    def __init__(
        self,
        control_points: torch.Tensor,   # (n_c, 3)
        W: torch.Tensor,                # (n_c, 3)
        A: torch.Tensor,                # (4, 3)
        *,
        kernel_eps: float = 0.0,
        device: Optional[torch.device] = None,
        use_double: bool = False,
        default_chunk_size: int = 200_000,
        cp_indices: Optional[torch.Tensor] = None,  # Original indices (for visualization to find target control points)
    ):
        self.control_points = control_points
        self.W = W
        self.A = A
        self.kernel_eps = kernel_eps
        self.device = device if device is not None else control_points.device
        self.dtype = torch.float64 if use_double else torch.float32
        self.default_chunk_size = default_chunk_size
        self.cp_indices = cp_indices  # Only for visualization: original target control points

    @torch.no_grad()
    def transform(
        self,
        points: torch.Tensor,
        *,
        chunk_size: Optional[int] = None,
        out_device: Optional[torch.device] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if points.numel() == 0:
            return points

        chunk = chunk_size or self.default_chunk_size
        dev = self.device
        dt = self.dtype

        pts = points.to(device=dev, dtype=dt, copy=False)
        cp  = self.control_points.to(device=dev, dtype=dt, copy=False)
        W   = self.W.to(device=dev, dtype=dt, copy=False)
        A   = self.A.to(device=dev, dtype=dt, copy=False)

        M = pts.shape[0]
        out = torch.empty((M, 3), device=dev, dtype=dt)

        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            pts_chunk = pts[s:e]
            m = pts_chunk.shape[0]

            affine = torch.cat([torch.ones(m, 1, device=dev, dtype=dt), pts_chunk], dim=1) @ A
            d_pc = torch.cdist(pts_chunk, cp)
            phi  = _kernel_linear_r(d_pc, eps=self.kernel_eps)
            nonlin = phi @ W
            out[s:e] = affine + nonlin

        if out_device is not None or out_dtype is not None:
            out = out.to(device=out_device or dev, dtype=out_dtype or dt)
        return out


@torch.no_grad()
def _fit_tps_3d(
    source: torch.Tensor,         # (N,3)
    target: torch.Tensor,         # (N,3)
    lambda_param: float,
    cp_idx: Optional[torch.Tensor],      # (n_c,) Data control point indices; None if using virtual control points
    device: Optional[torch.device],
    use_double: bool,
    kernel_eps: float,
    control_points: Optional[torch.Tensor] = None,  # (n_c,3) Virtual control points (e.g., voxel/leaf centers), can be None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      - W: (n_c,3)
      - A: (4,3)
      - cp_idx: Returns indices if using data control points; returns None if using virtual control points
    """
    assert source.shape == target.shape and source.shape[-1] == 3
    if device is None:
        device = source.device
    dtype = torch.float64 if use_double else torch.float32

    X = source.to(device=device, dtype=dtype, copy=False)   # (N,3)
    Y = target.to(device=device, dtype=dtype, copy=False)   # (N,3)

    # ---------------------------
    # Mode 1: Virtual control points (regression solution)
    # ---------------------------
    if control_points is not None:
        C = control_points.to(device=device, dtype=dtype, copy=False)  # (n_c,3)
        n_c = C.shape[0]
        N = X.shape[0]

        Phi = _kernel_linear_r(torch.cdist(X, C), eps=kernel_eps)      # (N, n_c)
        P_X = torch.cat([torch.ones(N, 1, device=device, dtype=dtype), X], dim=1)  # (N,4)

        # Ridge regularized least squares:  [Phi  P_X; sqrt(λ)I  0] [W;A] = [Y; 0]
        if lambda_param > 0:
            reg_left = torch.cat([
                torch.sqrt(torch.tensor(lambda_param, device=device, dtype=dtype)) *
                torch.eye(n_c, device=device, dtype=dtype),
                torch.zeros((n_c, 4), device=device, dtype=dtype)
            ], dim=1)                                                 # (n_c, n_c+4)
            LS = torch.cat([torch.cat([Phi, P_X], dim=1), reg_left], dim=0)  # (N+n_c, n_c+4)
            rhs = torch.cat([Y, torch.zeros((n_c, 3), device=device, dtype=dtype)], dim=0)  # (N+n_c,3)
        else:
            LS = torch.cat([Phi, P_X], dim=1)                          # (N, n_c+4)
            rhs = Y                                                    # (N,3)

        # Least squares solution
        try:
            sol, *_ = torch.linalg.lstsq(LS, rhs)   # (n_c+4, 3)
        except RuntimeError:
            sol = torch.pinverse(LS) @ rhs

        W = sol[:n_c, :]           # (n_c,3)
        A = sol[n_c:n_c+4, :]      # (4,3)
        return W, A, None          # Virtual control points have no cp_idx

    # ---------------------------
    # Mode 2: Data control points (interpolation solution, keep original logic)
    # ---------------------------
    assert cp_idx is not None and len(cp_idx) > 0, "cp_idx is None and control_points not provided"
    src_cp = X[cp_idx]   # (n_c,3)
    tgt_cp = Y[cp_idx]   # (n_c,3)
    n = src_cp.shape[0]

    # Perfect match → Identity
    if torch.allclose(src_cp, tgt_cp, atol=1e-6):
        W = torch.zeros((n, 3), device=device, dtype=dtype)
        A = torch.zeros((4, 3), device=device, dtype=dtype)
        A[1, 0] = 1.0; A[2, 1] = 1.0; A[3, 2] = 1.0
        return W, A, cp_idx

    # Classic TPS matrix system
    d_cc = torch.cdist(src_cp, src_cp)
    K = _kernel_linear_r(d_cc, eps=kernel_eps)
    P = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), src_cp], dim=1)

    L_top = torch.cat([K + lambda_param * torch.eye(n, device=device, dtype=dtype), P], dim=1)
    L_bottom = torch.cat([P.transpose(0, 1), torch.zeros((4, 4), device=device, dtype=dtype)], dim=1)
    L = torch.cat([L_top, L_bottom], dim=0)

    Ycp = torch.cat([tgt_cp, torch.zeros((4, 3), device=device, dtype=dtype)], dim=0)

    try:
        Xsol = torch.linalg.solve(L, Ycp)
    except RuntimeError:
        try:
            Xsol, *_ = torch.linalg.lstsq(L, Ycp)
        except RuntimeError:
            Xsol = torch.pinverse(L) @ Ycp

    W = Xsol[:n, :]
    A = Xsol[n:n+4, :]
    return W, A, cp_idx

def _visualize_o3d(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_tf: torch.Tensor,
    *,
    only_control_points: bool,
    model: TPS3DModel,
):
    import numpy as np
    import open3d as o3d

    def _to_np(t: torch.Tensor):
        x = t.detach().cpu()
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
        return x.numpy()

    if only_control_points:
        # Directly use the control points saved in the model and their mappings; target control points are taken from the original target using cp_indices
        cp_src = model.control_points
        cp_tf  = model.transform(cp_src)
        if model.cp_indices is not None:
            cp_tgt = tgt[model.cp_indices]
        else:
            # If indices are not saved, fallback to nearest neighbor matching (theoretically should not happen because we always save cp_indices)
            cp_tgt = cp_src  # Fallback to avoid errors

        src_np = _to_np(cp_src)
        tgt_np = _to_np(cp_tgt)
        tf_np  = _to_np(cp_tf)
    else:
        src_np = _to_np(src)
        tgt_np = _to_np(tgt)
        tf_np  = _to_np(src_tf)

    def make_pcd(pts_np, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float32), (pts_np.shape[0], 1))
        )
        return pcd

    pcd_src = make_pcd(src_np, (0.2, 0.5, 1.0))  # blue: src
    pcd_tgt = make_pcd(tgt_np, (1.0, 0.2, 0.2))  # red: tgt
    pcd_tf  = make_pcd(tf_np,  (0.2, 0.9, 0.2))  # green: src_transformed

    visl = o3d.visualization.Visualizer()
    visl.create_window("Blue: src, Red: tgt, Green: src transformed")
    for g in [pcd_src, pcd_tgt, pcd_tf]:
        visl.add_geometry(g)
    opt = visl.get_render_option()
    opt.point_size = 10
    visl.run()
    visl.destroy_window()


def fit_tps(
    src: torch.Tensor,            # (N,3)
    tgt: torch.Tensor,            # (N,3)
    *,
    cp_idx: Optional[torch.Tensor] = None,   # (n_c,)  Data control point indices; None if using virtual control points
    lambda_param: float = 1e-3,
    device: Optional[torch.device] = None,
    use_double: bool = False,
    chunk_size: int = 200_000,
    kernel_eps: float = 0.0,
    visualize: bool = True,
    visualize_only_control_points: bool = False,
) -> Tuple[TPS3DModel, torch.Tensor]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fit
    W, A, cp_idx = _fit_tps_3d(
        src, tgt,
        lambda_param=lambda_param,
        device=device,
        use_double=use_double,
        kernel_eps=kernel_eps,
        cp_idx=cp_idx
    )
    control_points = src.to(device)[cp_idx]

    # model
    model = TPS3DModel(
        control_points=control_points,
        W=W,
        A=A,
        kernel_eps=kernel_eps,
        device=device,
        use_double=use_double,
        default_chunk_size=chunk_size,
        cp_indices=cp_idx,   # Used for visualization to directly fetch corresponding control points from tgt
    )

    # Immediately transform src
    src_tf = model.transform(src)
    print(torch.norm(src_tf-src, dim=1).max())

    # Visualization
    if visualize:
        _visualize_o3d(
            src=src, tgt=tgt, src_tf=src_tf,
            only_control_points=visualize_only_control_points,
            model=model,
        )

    return model

def _to_h(P):
    P = np.asarray(P, dtype=np.float64)
    return np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)

def _transform_points(P_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    # return (_to_h(P_xyz) @ T.T)[:, :3]

    return (T @ _to_h(P_xyz).T).T[:, :3]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def robust_mean_func(points: np.ndarray,
                     method="mad",
                     thresh=3.5,
                     visualize: bool = False,
                     title: str = None,
                     save_path: str = None):
    """
    Remove outliers from a set of points before computing the mean, and visualize samples considered as outliers.

    Parameters:
        points: (M, 3) numpy array of 3D points
        method: "mad" | "std" | None/"None"
        thresh: Threshold factor
        visualize: Whether to perform 3D scatter visualization (inliers vs outliers)
        title: Visualization title (optional)
        save_path: If provided, saves the image to this path (e.g., "debug_outliers.png")

    Returns:
        mean: (3,) Filtered mean (or median/mean in extreme cases)
    """
    points = np.asarray(points)
    M = len(points)
    mask = np.ones(M, dtype=bool)

    # By default, assume all points are inliers

    if method is None or method == "None":
        mean = points.mean(axis=0)

    elif method == "mad":
        median = np.median(points, axis=0)
        dists = np.linalg.norm(points - median, axis=1)

        # MAD of distances
        med_d = np.median(dists)
        mad = np.median(np.abs(dists - med_d))

        if mad == 0:
            # All points are almost identical, keep all
            mask = np.ones(M, dtype=bool)
            mean = points.mean(axis=0)
        else:
            modified_z = 0.6745 * dists / mad
            mask = modified_z < thresh
            if np.any(mask):
                mean = points[mask].mean(axis=0)
            else:
                # If all points are considered outliers, fallback to median
                mask = np.ones(M, dtype=bool)
                mean = median

    elif method == "std":
        mean0 = points.mean(axis=0)
        dists = np.linalg.norm(points - mean0, axis=1)
        std = dists.std()

        if std == 0:
            mask = np.ones(M, dtype=bool)
            mean = mean0
        else:
            mask = dists < thresh * std
            if np.any(mask):
                mean = points[mask].mean(axis=0)
            else:
                # If all points are considered outliers, fallback to original mean
                mask = np.ones(M, dtype=bool)
                mean = mean0
    else:
        raise ValueError(f"Unknown method: {method}")

    # if visualize:
    #     _visualize_inliers_outliers(points, mask, title, save_path)

    return mean, mask

def set_axes_equal(ax):
    '''Manually set equal scaling for 3D axes to prevent matplotlib from automatic stretching'''
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)
    max_range = max(
        abs(xlim[1] - xlim[0]),
        abs(ylim[1] - ylim[0]),
        abs(zlim[1] - zlim[0])
    ) / 2

    ax.set_xlim3d([xmean - max_range, xmean + max_range])
    ax.set_ylim3d([ymean - max_range, ymean + max_range])
    ax.set_zlim3d([zmean - max_range, zmean + max_range])
    
def _visualize_inliers_outliers(points: np.ndarray,
                                mask: np.ndarray,
                                title: str = None,
                                save_path: str = None):
    """
    Internal utility function: 3D visualization of inliers and outliers.
    Inliers: dots
    Outliers: crosses
    """
    pts_in = points[mask]
    pts_out = points[~mask]

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    if len(pts_in) > 0:
        ax.scatter(pts_in[:, 0], pts_in[:, 1], pts_in[:, 2],
                   s=50, alpha=0.9, label=f"Inliers ({len(pts_in)})")
    if len(pts_out) > 0:
        ax.scatter(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2],
                   s=50, marker='x', alpha=0.9, label=f"Outliers ({len(pts_out)})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best")
    set_axes_equal(ax)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def avg_control_points(
    control_points: Dict[Any, Dict[Any, Dict[str, Any]]],
    submap_control_points: Dict[Any, List[int]],
    reference_homographys: Dict[Any, np.ndarray],
    points_in_world_frames, 
    colors,
    visualize: bool = False,
    point_size: float = 5.0,
    filter_method: str = None,
    filter_k: int = 32,
    robust_mean_method = None,
    poses = None,
    intrinsics = None,
    shapes = None,
    camera_color_map = None,
):
    """
    Display all point_ids in a single Open3D window:
      - Instances of each point_id (from different submaps, transformed to world) are shown as point clouds with unique colors
      - The mean of each point_id is marked with a larger sphere of the same color
    """
    print(len(control_points), "control points before filtering")
    del_keys = []
    for pid in control_points:
        if len(control_points[pid]) < 2:
            del_keys.append(pid)
        elif len(control_points[pid]) == 2:
            continue
        else:
            all_obs = np.array([control_points[pid][sid]['3d'] for sid in control_points[pid]])
            _, mask = robust_mean_func(all_obs, robust_mean_method, visualize=visualize)
            if np.sum(mask) < 2:
                del_keys.append(pid)
            else:
                control_points[pid] = {sid: control_points[pid][sid] for i, sid in enumerate(control_points[pid]) if mask[i]}
    for pid in del_keys:
        del control_points[pid]
    print(len(control_points), "control points after filtering")

    control_points_global = {pid: [] for pid in control_points.keys()}  # pid -> submap_id -> point
    submap_points_global = {}  # sid -> [points]
    control_points_count = [len(subdict) for subdict in control_points.values()]

    # if visualize:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 4))
    #     plt.hist(
    #         control_points_count,
    #         bins=np.arange(1, max(control_points_count) + 2, 1),  # 让每个整数落在柱子中心
    #         edgecolor='black',
    #         width=1.0
    #     )
    #     plt.xticks(range(1, max(control_points_count) + 1))  # X 轴刻度对齐整数下标
    #     plt.xlabel('Number of Submaps per Control Point')
    #     plt.ylabel('Count')
    #     plt.title('Histogram of Control Point Occurrences Across Submaps')
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.show()
        
    for submap_id, control_points_idx in submap_control_points.items():
        if len(control_points_idx) == 0:
            continue
        control_points_idx = [pid for pid in control_points_idx if pid in control_points and submap_id in control_points[pid]]
        control_points_local = [control_points[pid][submap_id]['3d'] for pid in control_points_idx]
        print(f"Submap {submap_id}: {len(control_points_local)} control points")
        control_points_world = _transform_points(np.asarray(control_points_local, dtype=np.float64).reshape(-1,3), reference_homographys[submap_id])
        submap_points_global[submap_id] = control_points_world
        for i, pid in enumerate(control_points_idx):
            if pid in control_points and submap_id in control_points[pid]:
                control_points_global[pid].append(control_points_world[i])
        submap_control_points[submap_id] = control_points_idx

    # control_points_global_mean = np.array([
    #     np.mean(np.asarray(control_points_global[pid]), axis=0)
    #     for pid in control_points_global
    # ])
    control_points_global_mean = {
        pid: np.mean(np.asarray(control_points_global[pid]), axis=0)
        for pid in control_points_global
    }
    submap_points_global_mean = {}
    for submap_id, control_points_idx in submap_control_points.items():
        if len(control_points_idx) == 0:
            continue
        # submap_points_global_mean[submap_id] = control_points_global_mean[np.array(control_points_idx)]
        submap_points_global_mean[submap_id] = np.array([control_points_global_mean[pid] for pid in control_points_idx])

    print(f"Total {len(control_points_global_mean)} unique control points")



    if visualize:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="All Control Points")
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.line_width = 100.0  
        opt.light_on = False

        state = {"step": False, "quit": False}
        vis.register_key_callback(ord("N"), lambda _: state.__setitem__("step", True))
        vis.register_key_callback(ord("Q"), lambda _: state.__setitem__("quit", True))

        first = True  # Reset bounding box on first frame, keep view for subsequent frames
    new_points_in_world_frames = {}
    for submap_id, submap_points in submap_points_global.items():
        if len(submap_points) == 0:
            continue
        submap_points_mean = submap_points_global_mean[submap_id]  # (N,3)
        assert submap_points.shape == submap_points_mean.shape and submap_points.shape[1] == 3
        N_local = submap_points.shape[0]
        print(f"Fitting TPS for submap {submap_id} with {N_local} control points")

        # random_colors = np.random.rand(len(submap_points_mean), 3)

        from matplotlib.colors import ListedColormap

        def colors_from_vector_directions(src, tgt, cmap_name='coolwarm', cut_mid=(0.45, 0.55), normalize=True):
            """
            Map colors based on the direction from src to tgt, skipping the whitish middle region of the 'coolwarm' colormap.
            Args:
                src, tgt: (N,3)
                cmap_name: Original colormap (default 'coolwarm')
                cut_mid: Middle range to exclude (default (0.45, 0.55))
                normalize: Whether to normalize the direction mapping values
            Returns:
                colors: (N,3)
            """
            src, tgt = np.asarray(src), np.asarray(tgt)
            flows = tgt - src
            norms = np.linalg.norm(flows, axis=1, keepdims=True) + 1e-12
            dirs = flows / norms

            # Encode direction using spherical coordinates
            theta = np.arccos(np.clip(dirs[:, 2], -1, 1))  # polar angle 0~pi
            phi = np.arctan2(dirs[:, 1], dirs[:, 0])       # azimuthal angle -pi~pi
            val = (theta / np.pi * 0.5 + (phi + np.pi) / (2 * np.pi) * 0.5)

            if normalize:
                val = (val - val.min()) / (val.max() - val.min() + 1e-8)

            # --- Construct a new colormap excluding the light middle region ---
            base_cmap = plt.get_cmap(cmap_name)
            lo, hi = cut_mid
            n_half = 128
            new_colors = np.vstack([
                base_cmap(np.linspace(0.0, lo, n_half)),
                base_cmap(np.linspace(hi, 1.0, n_half))
            ])
            cmap_trimmed = ListedColormap(new_colors)

            # To ensure uniform mapping, map val < 0.5 to the first half, and >0.5 to the second half
            val_adj = np.where(val < 0.5, val / 0.5 * 0.5, 0.5 + (val - 0.5) / 0.5 * 0.5)
            colors = cmap_trimmed(val_adj)[:, :3]
            return colors







        # def spheres_from_points(points, radius=0.007, colors=(1, 0, 0)):
        #     points = np.asarray(points, dtype=np.float64)
        #     colors = np.asarray(colors, dtype=float)
        #     per_point_color = (colors.ndim == 2 and colors.shape[0] == len(points))

        #     base = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        #     base.compute_vertex_normals()

        #     mesh = o3d.geometry.TriangleMesh()
        #     if per_point_color:
        #         for p, c in zip(points, colors):
        #             m = o3d.geometry.TriangleMesh(base)
        #             m.paint_uniform_color(c.tolist())
        #             m.translate(p)
        #             mesh += m
        #     else:
        #         base.paint_uniform_color(colors.tolist() if colors.ndim == 1 else (1, 0, 0))
        #         for p in points:
        #             m = o3d.geometry.TriangleMesh(base)
        #             m.translate(p)
        #             mesh += m
        #     return mesh
        

        def spheres_from_points(points,
                                                radius=0.007,
                                                colors=(1, 0, 0),
                                                voxel_size=0.1,
                                                voxel_color=(0, 0, 0),
                                                deduplicate=True):
            """
            Create:
            1) Small spheres at the center of the voxel each point belongs to (supports multiple colors)
            2) Wireframes of the corresponding voxels (not occluding the interior)

            Returns: (sphere_mesh, voxel_lines)
            """
            points = np.asarray(points, dtype=np.float64)
            if points.size == 0:
                return o3d.geometry.TriangleMesh(), o3d.geometry.LineSet()

            # Support anisotropic voxels
            if np.isscalar(voxel_size):
                vs = np.array([voxel_size, voxel_size, voxel_size], dtype=np.float64)
            else:
                vs = np.asarray(voxel_size, dtype=np.float64).reshape(3,)

            # Voxel indices and centers
            vox_idx = np.floor(points / vs).astype(np.int64)
            if deduplicate:
                uniq_keys, uniq_indices = np.unique(vox_idx, axis=0, return_index=True)
                centers = (uniq_keys.astype(np.float64) + 0.5) * vs
                color_src_indices = uniq_indices
            else:
                centers = (vox_idx.astype(np.float64) + 0.5) * vs
                color_src_indices = np.arange(len(points))

            # Sphere colors
            colors = np.asarray(colors, dtype=float)
            if colors.ndim == 1:
                sphere_colors = np.tile(colors[None, :], (len(centers), 1))
            else:
                sphere_colors = colors[color_src_indices]

            # Sphere mesh
            base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            base_sphere.compute_vertex_normals()
            sphere_mesh = o3d.geometry.TriangleMesh()
            for c, col in zip(centers, sphere_colors):
                m = o3d.geometry.TriangleMesh(base_sphere)
                m.paint_uniform_color(col.tolist())
                m.translate(c)
                sphere_mesh += m

            # --- Draw voxel wireframes ---
            all_points = []
            all_lines = []

            # Relative coordinates of the 8 corners of each cube
            cube_corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
            cube_corners = (cube_corners - 0.5) * vs

            # 12 edges
            edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                            [4, 5], [5, 6], [6, 7], [7, 4],
                            [0, 4], [1, 5], [2, 6], [3, 7]], dtype=int)

            for i, c in enumerate(centers):
                start_idx = len(all_points)
                all_points.extend((cube_corners + c).tolist())
                all_lines.extend((edges + start_idx).tolist())

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(all_points),
                lines=o3d.utility.Vector2iVector(all_lines)
            )
            line_colors = np.tile(np.asarray(voxel_color, dtype=float)[None, :], (len(all_lines), 1))
            line_set.colors = o3d.utility.Vector3dVector(line_colors)

            return [sphere_mesh, line_set]



        def _rotation_from_a_to_b(a, b):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            v = np.cross(a, b); c = np.dot(a, b)
            if c > 1.0 - 1e-12:
                return np.eye(3)
            if c < -1.0 + 1e-12:
                axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                v = np.cross(a, axis); v /= np.linalg.norm(v)
                K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                return np.eye(3) + 2*(K@K)
            K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            return np.eye(3) + K + K@K * ((1.0 - c)/(np.linalg.norm(v)**2))

        def arrow_line_from_points(origins,
                                targets=None,
                                flows=None,
                                sphere_radius=0.007,
                                length=0.03,
                                cyl_radius=None,
                                cone_radius=None,
                                cyl_ratio=0.3,
                                colors=(0.9, 0.2, 0.2),
                                normalize_if_targets=True,
                                scale=2,
                                join_eps_ratio=0.5,  # Key: overlap ratio at the joint
                                min_len=1e-6):
            """
            Generate arrows (cylinders + cones) starting from origins, with a slight overlap between the cylinder and cone controlled by join_eps_ratio to avoid seams.
            - If targets or flows are provided, use their directions; otherwise, point along +Z with length=length.
            - The starting point is on the sphere surface: origin + sphere_radius * dir.
            - join_eps_ratio: a small overlap ratio relative to the total length L_full (default 1e-3).
            """
            origins = np.asarray(origins, dtype=np.float64)

            if flows is None and targets is not None:
                flows = np.asarray(targets, dtype=np.float64) - origins
            elif flows is not None:
                flows = np.asarray(flows, dtype=np.float64)
            # flows *= 100

            N = len(origins)
            z = np.array([0.0, 0.0, 1.0])

            colors = np.asarray(colors, dtype=float)
            if colors.ndim == 1:
                colors = np.tile(colors[None, :], (N, 1))
            assert colors.shape == (N, 3)

            if cyl_radius is None:
                cyl_radius = sphere_radius * 0.4
            if cone_radius is None:
                cone_radius = sphere_radius * 0.8

            mesh = o3d.geometry.TriangleMesh()

            for i in range(N):
                p0 = origins[i]

                if flows is None:
                    dir_vec = z.copy()
                    L_full = max(length * scale, min_len)
                else:
                    f = flows[i]; Lf = np.linalg.norm(f)
                    if Lf < min_len: continue
                    dir_vec = f / Lf
                    L_full = max((length if normalize_if_targets else Lf) * scale, min_len)

                base = p0 + sphere_radius * dir_vec

                # Slight overlap at the joint (to fix numerical errors/shading seams)
                join_eps = max(L_full * join_eps_ratio, 1e-9)
                L_cyl = max(L_full * cyl_ratio + join_eps, min_len)  # Cylinder slightly longer
                L_cone = max(L_full - (L_full * cyl_ratio), min_len * 0.5)

                # Cylinder (bottom z=0 top z=L_cyl), rotate to dir and place at base
                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=L_cyl)
                cyl.compute_vertex_normals()
                cyl.paint_uniform_color(colors[i].tolist())
                R = _rotation_from_a_to_b(z, dir_vec)
                cyl.rotate(R, center=(0, 0, 0))
                cyl.translate(base, relative=True)

                # Cone (bottom z=0 top z=L_cone), place at the theoretical joint base + (L_cyl - join_eps) * dir
                # Since L_cyl already includes +join_eps, pull the cone back by join_eps for slight overlap
                cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=L_cone)
                cone.compute_vertex_normals()
                cone.paint_uniform_color(colors[i].tolist())
                cone.rotate(R, center=(0, 0, 0))
                cone.translate(base + (L_cyl - join_eps) * dir_vec, relative=True)

                mesh += cyl
                mesh += cone

            return mesh



        point_radius = 0.03

        if visualize:
            geoms = []
            if poses is not None:
                # Camera poses
                for i, pose in enumerate(poses[submap_id]):
                    if i % len(intrinsics) > 2:
                        continue
                    if i // len(intrinsics) < 2:
                        color = camera_color_map[0]
                    else:
                        color = camera_color_map[1]
                    geoms.extend(create_camera_frustum(pose, intrinsics[i%len(intrinsics)], shapes[i%len(intrinsics)], color, 1.0/10))
                

            # Background/submap point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frames[submap_id])
            pcd.colors = o3d.utility.Vector3dVector(colors[submap_id])
            geoms.append(pcd)

            # if N_local == 0:
            #     # Even if there are no control points, refresh the background
            #     vis.clear_geometries()
            #     for g in geoms:
            #         vis.add_geometry(g, reset_bounding_box=first)
            #     vis.poll_events(); vis.update_renderer()
            #     first = False
            #     continue

            # Submap mean points
            # pcd_sub = o3d.geometry.PointCloud()
            # pcd_sub.points = o3d.utility.Vector3dVector(submap_points)
            # pcd_sub.colors = o3d.utility.Vector3dVector(random_colors)
            # geoms.append(pcd_sub)
            
            # Highlight: corresponding points & mean points → spheres (radius can be controlled independently)


            # —— Refresh the geometries for this round —— #
            vis.clear_geometries()
            for g in geoms + spheres_from_points(submap_points,      radius=point_radius, colors=colors_from_vector_directions(submap_points, submap_points_mean)):
                vis.add_geometry(g, reset_bounding_box=first)
            vis.poll_events()
            vis.update_renderer()
            first = False

            state["step"] = False
            # print(f"[Submap {submap_id}] Press 'N' for next, 'Q' to quit.")
            while not state["step"] and not state["quit"]:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.02)
            if state["quit"]:
                visualize = False
                vis.destroy_window()
                
        if visualize:

            # # Submap mean points
            # pcd_sub = o3d.geometry.PointCloud()
            # pcd_sub.points = o3d.utility.Vector3dVector(submap_points)
            # pcd_sub.colors = o3d.utility.Vector3dVector(random_colors)
            # geoms.append(pcd_sub)
            # pcd_sub = o3d.geometry.PointCloud()
            # pcd_sub.points = o3d.utility.Vector3dVector(submap_points_mean)
            # pcd_sub.colors = o3d.utility.Vector3dVector(random_colors)
            # geoms.append(pcd_sub)

            # # Lines (corresponding points -> mean points)
            # line_pts = np.vstack([submap_points, submap_points_mean])
            # lines = np.array([[i, i + N_local] for i in range(N_local)], dtype=np.int32)
            # line_set = o3d.geometry.LineSet(
            #     points=o3d.utility.Vector3dVector(line_pts),
            #     lines=o3d.utility.Vector2iVector(lines),
            # )
            # line_set.colors = pcd_sub.colors
            # geoms.append(line_set)

            


            # Lines (corresponding points -> mean points)
            # line_pts = np.vstack([submap_points, submap_points_mean])
            # lines = np.array([[i, i + N_local] for i in range(N_local)], dtype=np.int32)
            # line_set = o3d.geometry.LineSet(
            #     points=o3d.utility.Vector3dVector(line_pts),
            #     lines=o3d.utility.Vector2iVector(lines),
            # )
            # line_set.colors = o3d.utility.Vector3dVector(random_colors)
            # geoms.append(line_set)

            # —— Refresh the geometries for this round —— #
            vis.clear_geometries()
            this_colors = colors_from_vector_directions(submap_points, submap_points_mean)
            for g in geoms+spheres_from_points(submap_points,      radius=point_radius, colors=this_colors) + [
                            arrow_line_from_points(submap_points, targets=submap_points_mean, colors=this_colors, sphere_radius=point_radius)]:
                vis.add_geometry(g, reset_bounding_box=first)
            vis.poll_events()
            vis.update_renderer()
            first = False  # Do not reset the view afterwards

            # —— Blocking wait (press N to continue / Q to quit), can be removed if not needed —— #
            state["step"] = False
            # print(f"[Submap {submap_id}] Press 'N' for next, 'Q' to quit.")
            while not state["step"] and not state["quit"]:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.02)
            if state["quit"]:
                visualize = False
                vis.destroy_window()
            
            
        submap_points_mean = filter_control_points(
            src_cp=torch.tensor(submap_points, device='cuda'),
            tgt_cp=torch.tensor(submap_points_mean, device='cuda'),
            method=filter_method,
            k=filter_k,
        ).cpu().numpy()

        if visualize:
            # geoms = []
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_in_world_frames[submap_id])
            # pcd.colors = o3d.utility.Vector3dVector(colors[submap_id])
            # geoms.append(pcd)
            # pcd_sub = o3d.geometry.PointCloud()
            # pcd_sub.points = o3d.utility.Vector3dVector(submap_points_mean)
            # pcd_sub.colors = o3d.utility.Vector3dVector(random_colors)
            # geoms.append(pcd_sub)

            # # Lines (corresponding points -> mean points)
            # line_pts = np.vstack([submap_points, submap_points_mean])
            # lines = np.array([[i, i + N_local] for i in range(N_local)], dtype=np.int32)
            # line_set = o3d.geometry.LineSet(
            #     points=o3d.utility.Vector3dVector(line_pts),
            #     lines=o3d.utility.Vector2iVector(lines),
            #     # colors=o3d.utility.Vector3dVector([[230/255,109/255,80/255] for _ in range(len(lines))])
            # )
            # line_set.colors = pcd_sub.colors
            # geoms.append(line_set)


            # —— Refresh the geometries for this round —— #
            vis.clear_geometries()
            for g in geoms+[arrow_line_from_points(submap_points, targets=submap_points_mean, colors=colors_from_vector_directions(submap_points, submap_points_mean), sphere_radius=point_radius)]:
                vis.add_geometry(g, reset_bounding_box=first)
            vis.poll_events()
            vis.update_renderer()
            first = False  # Do not reset the view afterwards

            state["step"] = False
            # print(f"[Submap {submap_id}] Press 'N' for next, 'Q' to quit.")
            while not state["step"] and not state["quit"]:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.02)
            if state["quit"]:
                visualize = False
                vis.destroy_window()

        num = len(submap_points_global.get(submap_id, []))
        if num == 0:
            print(f"Not enough control points for submap {submap_id} to fit TPS")
            continue
        src = submap_points
        tgt = submap_points_mean
        model = fit_tps(src=torch.tensor(src, device='cuda'), tgt=torch.tensor(tgt, device='cuda'), cp_idx=list(range(num)), visualize=False)
        points_in_world = points_in_world_frames[submap_id]
        points_in_world = model.transform(torch.tensor(points_in_world, device='cuda')).cpu().numpy()

        new_points_in_world_frames[submap_id] = points_in_world

        if visualize:
            geoms = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world)
            pcd.colors = o3d.utility.Vector3dVector(colors[submap_id])
            geoms.append(pcd)
            
            
            vis.clear_geometries()
            for g in geoms:
                vis.add_geometry(g, reset_bounding_box=first)
            vis.poll_events()
            vis.update_renderer()

            state["step"] = False
            # print(f"[Submap {submap_id}] Press 'N' for next, 'Q' to quit.")
            while not state["step"] and not state["quit"]:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.02)
            if state["quit"]:
                visualize = False
                vis.destroy_window()

    if visualize:
        vis.destroy_window()
    
    return new_points_in_world_frames

@torch.no_grad()
def filter_control_points(
    src_cp: torch.Tensor,              # (N,3)
    tgt_cp: torch.Tensor,              # (N,3)
    *,
    method: str = "gaussian",            # "mean" | "gaussian" | "robust"
    k: int = 64,                       # KNN neighbors (including itself)
    sigma: Optional[float] = None,     # Gaussian kernel standard deviation (if None, estimated automatically)
    alpha: float = 2.0,                # Direction consistency exponent (robust)
    beta_percentile: float = 0.75,     # Displacement magnitude robust scale percentile (robust)
    angle_thresh_deg: Optional[float] = None,  # If set, neighborhoods with angle > threshold are assigned minimal weight
    iters: int = 1,                    # Number of iterations (>1 for smoother results)
    blend: float = 1.0,                # 0..1, 1=use fully smoothed result; <1 interpolates with original displacement
    knn_chunk: Optional[int] = None,   # KNN chunk size; None means all at once (convenient for small N)
) -> torch.Tensor:
    """
    Returns: tgt_cp_filtered (N,3)

    Description:
      - Filters the displacement Δ = tgt_cp - src_cp;
      - Neighborhood is based on KNN of src_cp;
      - method:
        * "mean"     : uniform weights
        * "gaussian" : w = exp(-||xi-xj||^2 / (2σ^2))
        * "robust"   : w = w_space * (max(0,cos))^alpha * 1/(1+||Δj||/β)
      - sigma is estimated automatically: if None, the median KNN distance is used as the scale (more robust)
      - angle_thresh_deg: if given (e.g., 60), neighborhoods with angle > threshold have their direction weights set to minimal
      - iters>1 applies the filtering iteratively using the previous iteration's Δ_smooth
      - blend < 1: Δ_out = (1-blend)*Δ_orig + blend*Δ_smooth
    """
    assert src_cp.shape == tgt_cp.shape and src_cp.shape[-1] == 3
    device = src_cp.device
    dtype = src_cp.dtype

    N = src_cp.shape[0]
    if N == 0 or method is None or method == "None":
        return tgt_cp

    # --------- KNN indices (row chunking to avoid OOM for large NxN) ----------
    def knn_indices(x: torch.Tensor, k: int, chunk: Optional[int]) -> torch.Tensor:
        # Returns (N,k) neighbor indices, including self (usually self is the closest)
        if chunk is None:
            d2 = torch.cdist(x, x, p=2)**2  # (N,N)
            idx = torch.topk(-d2, k=k, dim=1).indices  # Negative sign → take smallest distances
            return idx
        else:
            idx_all = torch.empty((N, k), device=device, dtype=torch.long)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                d2_block = torch.cdist(x[s:e], x, p=2)**2  # (b,N)
                idx_block = torch.topk(-d2_block, k=k, dim=1).indices
                idx_all[s:e] = idx_block
            return idx_all

    # Initial displacement
    Delta = (tgt_cp - src_cp).to(dtype=dtype, device=device)
    X = src_cp.to(dtype=dtype, device=device)

    # Precompute KNN
    k = min(k, max(1, N))  # clamp
    nbr_idx = knn_indices(X, k=k, chunk=knn_chunk)  # (N,k)

    # Automatic sigma (based on the median distance to the k-1-th neighbor; valid when k>=2)
    if sigma is None:
        with torch.no_grad():
            # Extract neighborhoods and compute the distance from each i to its k-th neighbor
            Xi = X[:, None, :].expand(-1, k, -1)          # (N,k,3)
            Xj = X[nbr_idx]                                # (N,k,3)
            d2 = ((Xi - Xj) ** 2).sum(dim=-1)             # (N,k)
            if k > 1:
                kth = torch.sqrt(torch.gather(d2, 1, torch.full((N,1), k-1, device=device, dtype=torch.long)))
                sigma = float(torch.median(kth).item() + 1e-12)
            else:
                # Cannot estimate; fallback to bounding box diagonal ratio
                mins = X.min(0).values; maxs = X.max(0).values
                sigma = float((maxs - mins).norm().item()) * 1e-3 + 1e-9

    # Robust scale β (used in "robust" method)
    if method == "robust":
        mags = Delta.norm(dim=-1)
        beta = torch.quantile(mags, q=torch.tensor(beta_percentile, device=device)).item() + 1e-12
        cos_clip_min = 0.0
        if angle_thresh_deg is not None:
            # Angle threshold: cos θ, direction weights ≈ 0 if below this value
            import math
            cos_clip_min = max(0.0, float(torch.cos(torch.tensor(angle_thresh_deg * math.pi / 180.0)).item()))

    # Iterative filtering
    for _ in range(max(1, iters)):
        # Neighborhood data
        Dj = Delta[nbr_idx]                  # (N,k,3)
        Xi = X[:, None, :].expand(-1, k, -1) # (N,k,3)
        Xj = X[nbr_idx]                      # (N,k,3)
        d2 = ((Xi - Xj) ** 2).sum(dim=-1)    # (N,k)

        # —— Weights —— 
        if method == "mean":
            w = torch.ones((N, k), device=device, dtype=dtype)

        elif method == "gaussian":
            w = torch.exp(-d2 / (2.0 * (sigma**2)))

        elif method == "robust":
            # Spatial Gaussian
            w_space = torch.exp(-d2 / (2.0 * (sigma**2)))
            # Direction consistency: cos(Delta_i, Delta_j) ∈ [-1,1] → clamp
            Di = Delta[:, None, :]                       # (N,1,3)
            # Avoid NaN caused by zero vectors
            eps = 1e-12
            cos = torch.nn.functional.cosine_similarity(
                Di.clamp_min(-eps).clamp_max(eps) + Di,  # trick to avoid all-zero vectors
                Dj.clamp_min(-eps).clamp_max(eps) + Dj,
                dim=-1
            )
            if angle_thresh_deg is not None:
                cos = torch.where(cos >= cos_clip_min, cos, torch.full_like(cos, 0.0))
            w_dir = torch.clamp(cos, min=0.0) ** alpha   # Only same or similar directions are effective

            # Magnitude robustness: 1 / (1 + ||Δj||/β)
            w_mag = 1.0 / (1.0 + (Dj.norm(dim=-1) / beta))

            w = (w_space * w_dir * w_mag).clamp_min(1e-12)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize and compute weighted average
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)   # (N,1)
        Delta_smooth = (w[..., None] * Dj).sum(dim=1) / w_sum  # (N,3)

        # Linear interpolation with original displacement
        Delta = (1.0 - blend) * Delta + blend * Delta_smooth

    tgt_cp_filtered = src_cp + Delta
    return tgt_cp_filtered




if __name__ == "__main__":
    pass