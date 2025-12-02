import json
import os
from os.path import join
import cv2
import ipdb
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import argparse
import glob
from termcolor import colored
from pykdtree.kdtree import KDTree as pyKDTree
from matplotlib import pyplot as plt


camera_color_map_nuscenes = {
    'CAM_FRONT': [0.106, 0.620, 0.467],  # 蓝绿 #1b9e77
    'CAM_BACK_RIGHT': [0.400, 0.651, 0.118],  # 深绿 #66a61e
    'CAM_BACK_LEFT': [0.914, 0.541, 0.765],  # 粉紫 #e7298a
    'CAM_FRONT_LEFT': [0.851, 0.373, 0.008],  # 橙棕 #d95f02
    'CAM_FRONT_RIGHT': [0.459, 0.439, 0.702],  # 紫 #7570b3
    'CAM_BACK': [0.901, 0.670, 0.008],  # 黄褐 #e6ab02
}
camera_color_map_waymo = {
    'FRONT': [0.106, 0.620, 0.467],  # 蓝绿 #1b9e77
    'FRONT_LEFT': [0.400, 0.651, 0.118],  # 深绿 #66a61e
    'FRONT_RIGHT': [0.914, 0.541, 0.765],  # 粉紫 #e7298a
    'SIDE_RIGHT': [0.851, 0.373, 0.008],  # 橙棕 #d95f02
    'SIDE_LEFT': [0.459, 0.439, 0.702],  # 紫 #7570b3
}

camera_color_map = [
    [0.106, 0.620, 0.467],  # 蓝绿 #1b9e77
    [0.400, 0.651, 0.118],  # 深绿 #66a61e
    [0.914, 0.541, 0.765],  # 粉紫 #e7298a
    
    # [0.122, 0.471, 0.706],  # 蓝 #1f78b4
    # [0.698, 0.874, 0.541],  # 浅绿 #b2df8a
    # [1.000, 0.498, 0.055],  # 橙 #ff7f0e
    # [0.890, 0.102, 0.110],  # 红 #e31a1c
    # [0.596, 0.306, 0.639],  # 紫 #6a3d9a
    # [0.651, 0.808, 0.890],  # 浅蓝 #a6cee3
]

cam_list_nuscenes = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]
cam_list_waymo = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


def create_camera_frustum(pose, K, img_size, color=[1, 0, 0], frustum_length=0.2, draw_axis=False):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = img_size
    pts_img = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    pts_norm = np.stack([(pts_img[:, 0] - cx) / fx, (pts_img[:, 1] - cy) / fy, np.ones(4)], axis=1)
    pts_norm *= frustum_length
    origin = np.zeros((1, 3))
    vertices = np.vstack([origin, pts_norm])  # 5x3

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = np.tile(np.array(color)[None], (len(lines), 1))
    lineset.colors = o3d.utility.Vector3dVector(colors)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(pts_norm)
    plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane.paint_uniform_color(color)
    plane.compute_vertex_normals()

    plane_back = o3d.geometry.TriangleMesh()
    plane_back.vertices = o3d.utility.Vector3dVector(pts_norm)
    plane_back.triangles = o3d.utility.Vector3iVector([[2, 1, 0], [3, 2, 0]])
    plane_back.paint_uniform_color(color)
    plane_back.compute_vertex_normals()

    plane.transform(pose)
    plane_back.transform(pose)
    lineset.transform(pose)

    if draw_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frustum_length)
        axis.transform(pose)
        return lineset, plane, plane_back, axis
    else:
        return lineset, plane, plane_back

def construct_GT_points(scene_path, 
           dataset="nuscenes",
           cam_list=None,
           ego_range = {
                'w': 1.73,
                'l': 4.08,
                'h': 1.84,
                'eps': 0.1,
            },
           pred_scene_path=None
           ):
    if os.path.exists(f"{scene_path}/GT_points.npz"):
        print(f"Loading existing GT points from {scene_path}/GT_points.npz")
        data = np.load(f"{scene_path}/GT_points.npz")
        all_points = data['points']
        all_colors = data['colors']
        lidar_on_image_mask_ls = []
        return all_points, all_colors, lidar_on_image_mask_ls
    pred_scene_path = None
    zero_pad = 3
    frame_num = len(glob.glob(join(scene_path, "lidar", "*.bin")))
    intrinsic3x3_dict = {cam: np.loadtxt(join(scene_path, f"intrinsic/{cam}.txt")) for cam in cam_list}
    all_points = []
    all_colors = []
    lidar_on_image_mask_ls = []
    for frame in range(frame_num):
        if dataset == 'nuscenes':
            lidar2world = np.loadtxt(join(scene_path, f"lidar/{frame:0>{zero_pad}d}.txt"))
            xyz_lidar = np.fromfile(join(scene_path, f"lidar/{frame:0>{zero_pad}d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
            w, l, h, slack = ego_range['w'], ego_range['l'], ego_range['h'], ego_range['eps']        
            inside_ego_range = np.array([
                [-w/2-slack, -l*0.4-slack, -h-slack],
                [w/2+slack, l*0.6+slack, slack]
            ])
            inside_mask = (xyz_lidar >= inside_ego_range[0:1]).all(1) & (xyz_lidar <= inside_ego_range[1:]).all(1) 
            xyz_lidar = xyz_lidar[~inside_mask]
            xyz_world = (lidar2world[:3, :3] @ xyz_lidar.T + lidar2world[:3, 3:4]).T
        else:
            xyz_world = np.fromfile(join(scene_path, f"lidar/{frame:0>{zero_pad}d}.bin"), dtype=np.float32).reshape(-1, 3)
        
        N = xyz_world.shape[0]
        point_rgb_sum = np.zeros((N, 3), dtype=np.uint8)
        point_count = np.zeros((N), dtype=np.uint8)
        for cam in cam_list:
            image_path = join(scene_path, f"image/{cam}/{frame:0>{zero_pad}d}.jpg")
            assert os.path.exists(image_path), f"Image not found: {image_path}"
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]
            lidar_on_image_mask = np.zeros(image.shape[:2], dtype=bool)
            if pred_scene_path is not None:
                pred_valid_mask = np.load(join(pred_scene_path, f"{cam}/{frame:0>{zero_pad}d}.npz"))['mask'].astype(np.uint8)
                pred_valid_mask = cv2.resize(pred_valid_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                pred_valid_mask = None
                

            cam2world = np.loadtxt(join(scene_path, f"cam2world/{cam}/{frame:0>{zero_pad}d}.txt"))
            world2cam = np.linalg.inv(cam2world)
            R_w2c = world2cam[:3, :3]
            T_w2c = world2cam[:3, 3]
            ###########################################################################
            xyz_camera = (R_w2c @ xyz_world.T + T_w2c.reshape(3, 1)).T
            xyz_image = (intrinsic3x3_dict[cam] @ xyz_camera.T).T
            depth = xyz_image[:, 2]
            uv_image = xyz_image[:, :2] / xyz_image[:, 2:3]
            valid_mask = (depth > 0) \
                & (uv_image[:, 0] >= 0) \
                    & (uv_image[:, 0] < width) \
                        & (uv_image[:, 1] >= 0) \
                            & (uv_image[:, 1] < height)
            pixel_coords = uv_image[valid_mask].astype(int)
            pixel_coords = np.concatenate([pixel_coords, pixel_coords+np.array([[1,0]]), pixel_coords+np.array([[0,1]]), pixel_coords+np.array([[1,1]])], axis=0)
            pixel_coords = pixel_coords[pixel_coords[:,0]<width]
            pixel_coords = pixel_coords[pixel_coords[:,1]<height]
            lidar_on_image_mask[pixel_coords[:, 1], pixel_coords[:, 0]] = True
            lidar_on_image_mask_ls.append(lidar_on_image_mask)
            if pred_valid_mask is not None:
                uv_int = np.round(uv_image[valid_mask]).astype(int)
                mask_values = pred_valid_mask[np.clip(uv_int[:, 1], 0, height - 1), np.clip(uv_int[:, 0], 0, width - 1)]
                valid_mask_indices = np.where(valid_mask)[0]
                valid_mask[valid_mask_indices[mask_values == 0]] = False

            uv_image = uv_image[valid_mask].astype(np.float32)
            if np.count_nonzero(valid_mask) > 0:
                point_count[valid_mask] += 1
                point_rgb_sum[valid_mask] += cv2.remap(image, uv_image[:, 0], uv_image[:, 1], interpolation=cv2.INTER_LINEAR).reshape(-1, 3)

        xyz_world = xyz_world[point_count > 0]
        point_rgb_sum = point_rgb_sum[point_count > 0]
        point_count = point_count[point_count > 0]
        point_rgb = point_rgb_sum / point_count[:, None]

        all_points.append(xyz_world)
        all_colors.append(point_rgb)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)/ 255.0
    # print(f"Total points: {len(all_points)}")
    np.savez(f"{scene_path}/GT_points.npz", points=all_points, colors=all_colors)
    return all_points, all_colors, lidar_on_image_mask_ls

def vis_GT(gt_path, pred_path, dataset="nuscenes"):
    zero_pad = 3
    if dataset == "nuscenes":
        cam_list=cam_list_nuscenes
        camera_color_map = camera_color_map_nuscenes
    else:
        cam_list=cam_list_waymo
        camera_color_map = camera_color_map_waymo
    cam_list = [cam for cam in cam_list if os.path.exists(join(pred_path, cam))]
    frame_num = len(glob.glob(join(gt_path, "lidar", "*.bin")))
    intrinsic3x3_dict = {cam: np.loadtxt(join(gt_path, f"intrinsic/{cam}.txt")) for cam in cam_list}
    all_points, all_colors, lidar_on_image_mask_ls = construct_GT_points(gt_path, dataset=dataset, cam_list=cam_list)
    print(f"Total GT points: {len(all_points)}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    geometries = [pcd]

    for frame in range(frame_num):
        for cam in cam_list:
            image_path = join(gt_path, f"image/{cam}/{frame:0>{zero_pad}d}.jpg")
            assert os.path.exists(image_path), f"Image not found: {image_path}"
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]

            cam2world = np.loadtxt(join(gt_path, f"cam2world/{cam}/{frame:0>{zero_pad}d}.txt"))
            geometries.extend(create_camera_frustum(cam2world, intrinsic3x3_dict[cam], (height, width), camera_color_map[cam], 1.5))

    return geometries

def vis_pred(gt_path, pred_path, dataset="nuscenes"):
    if dataset == "nuscenes":
        camera_color_map = camera_color_map_nuscenes
        cam_list=cam_list_nuscenes
    else:
        camera_color_map = camera_color_map_waymo
        cam_list=cam_list_waymo
    cam_list = [cam for cam in cam_list if os.path.exists(join(pred_path, cam))]
    intrinsic3x3_dict = {cam: np.loadtxt(join(gt_path, f"intrinsic/{cam}.txt")) for cam in cam_list}

    geometries = []
    if 'tps' in pred_path and os.path.exists(f"{pred_path}/tps_fitted_points.npz"):
        print("Loading TPS fitted points...")
        data = np.load(f"{pred_path}/tps_fitted_points.npz")
        all_points = data['world_points_tps']
        all_colors = data['colors']/255
        frame_num = len(glob.glob(join(pred_path, f"{cam_list[0]}/*.npz")))
        for frame in range(frame_num):
            for cam in cam_list:
                image_path = join(gt_path, f"image/{cam}/{frame:0>3d}.jpg")
                assert os.path.exists(image_path), f"Image not found: {image_path}"
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height = image.shape[1], image.shape[0]
                pcd = np.load(f"{pred_path}/{cam}/{frame:0>3d}.npz")
                cam2world = pcd['pose']
                geometries.extend(create_camera_frustum(cam2world, intrinsic3x3_dict[cam], (height, width), camera_color_map[cam], 1.0/10))
    else:
        all_points = []
        all_colors = []
        frame_num = len(glob.glob(join(pred_path, f"{cam_list[0]}/*.npz")))
        for frame in range(frame_num):
            for cam in cam_list:
                image_path = join(gt_path, f"image/{cam}/{frame:0>3d}.jpg")
                assert os.path.exists(image_path), f"Image not found: {image_path}"
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height = image.shape[1], image.shape[0]

                # cam2world = np.loadtxt(f"{pred_path}/{cam}/{frame:0>{zero_pad}d}.txt")
                pcd = np.load(f"{pred_path}/{cam}/{frame:0>3d}.npz")
                all_points.append(pcd['point'][pcd['mask']])
                all_colors.append(pcd['color'][pcd['mask']])
                cam2world = pcd['pose']

                geometries.extend(create_camera_frustum(cam2world, intrinsic3x3_dict[cam], (height, width), camera_color_map[cam], 1.0/10))
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)/ 255.0

    # submap_num = len(glob.glob(f"{pred_path}/*.npz"))
    # print(f"Found {submap_num} submaps.")
    # for submap in range(submap_num):
    #     pcd = np.load(f"{pred_path}/{submap:0>3d}.npz")
    #     xyz_world = pcd['pcd']
    #     point_rgb = pcd['color']
    #     all_points.append(xyz_world)
    #     all_colors.append(point_rgb)

    # mean = np.median(all_points, axis=0)
    # valid_mask = np.linalg.norm(all_points - mean[None, :], axis=1) < 50.0
    # # valid_mask = 
    # all_points = all_points[valid_mask]
    # all_colors = all_colors[valid_mask]

    print(f"Total pred points: {len(all_points)}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    geometries.append(pcd)

    return geometries

def load_pred(pred_path, lidar_on_image_mask_ls=[], load_pcd=False):
    if 'nuscenes' in pred_path:
        cam_list = cam_list_nuscenes
    else:
        cam_list = cam_list_waymo
    cam_list = [cam for cam in cam_list if os.path.exists(join(pred_path, cam))]
    # if load_pcd:
    #     submap_num = len(glob.glob(f"{scene_path}/*.npz"))
    #     for submap in range(submap_num):
    #         pcd = np.load(f"{scene_path}/{submap:0>3d}.npz")
    #         xyz_world = pcd['pcd']
    #         point_rgb = pcd['color']
    #         all_points.append(xyz_world)
    #         all_colors.append(point_rgb)
    if not load_pcd:
        cam2worlds = [[] for _ in cam_list]
        frame_num = len(glob.glob(join(pred_path, f"{cam_list[0]}/*.npz")))
        for frame in range(frame_num):
            for i, cam in enumerate(cam_list):
                pcd = np.load(f"{pred_path}/{cam}/{frame:0>3d}.npz")
                cam2worlds[i].append(pcd['pose'])
        return None, [np.stack(cam2world, axis=0) for cam2world in cam2worlds]
    
    if 'tps' in pred_path and os.path.exists(f"{pred_path}/tps_fitted_points.npz"):
        print("Loading TPS fitted points...")
        data = np.load(f"{pred_path}/tps_fitted_points.npz")
        all_points = data['world_points_tps']
        all_colors = data['colors']/255
        cam2worlds = [[] for _ in cam_list]
        frame_num = len(glob.glob(join(pred_path, f"{cam_list[0]}/*.npz")))
        for frame in range(frame_num):
            for i, cam in enumerate(cam_list):
                pcd = np.load(f"{pred_path}/{cam}/{frame:0>3d}.npz")
                cam2worlds[i].append(pcd['pose'])
    else:
        all_points = []
        all_colors = []
        all_masks = [[] for _ in cam_list]
        cam2worlds = [[] for _ in cam_list]
        # ipdb.set_trace()
        frame_num = len(glob.glob(join(pred_path, f"{cam_list[0]}/*.npz")))
        for frame in range(frame_num):
            for i, cam in enumerate(cam_list):
                pcd = np.load(f"{pred_path}/{cam}/{frame:0>3d}.npz")
                mask = pcd['mask']
                if len(lidar_on_image_mask_ls) > 0:
                    lidar_on_image_mask = lidar_on_image_mask_ls[frame * len(cam_list) + i].astype(np.uint8)
                    lidar_on_image_mask = cv2.resize(lidar_on_image_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask = mask & lidar_on_image_mask
                # ipdb.set_trace()
                all_points.append(pcd['point'][mask])
                all_colors.append(pcd['color'][mask])
                # all_masks[i].append(pcd['mask'])
                cam2worlds[i].append(pcd['pose'])
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)/ 255.0


    print(f"Total pred points: {len(all_points)}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)


    return pcd, [np.stack(cam2world, axis=0) for cam2world in cam2worlds]

def load_GT(gt_path, pred_path, load_pcd=False):
    if 'nuscenes' in gt_path:
        cam_list = cam_list_nuscenes
    else:
        cam_list = cam_list_waymo
    cam_list = [cam for cam in cam_list if os.path.exists(join(pred_path, cam))]
    frame_num = len(glob.glob(join(gt_path, "lidar", "*.bin")))
    cam2worlds = [[] for _ in cam_list]
    for frame in range(frame_num):
        for i, cam in enumerate(cam_list):
            cam2world = np.loadtxt(f"{gt_path}/cam2world/{cam}/{frame:0>3d}.txt")
            cam2worlds[i].append(cam2world)
    
    if load_pcd:
        all_points, all_colors, lidar_on_image_mask_ls = construct_GT_points(gt_path, dataset='nuscenes' if 'nuscenes' in gt_path else 'waymo', cam_list=cam_list)
        print(f"Total GT points: {len(all_points)}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    else:
        pcd = None
        lidar_on_image_mask_ls = []

    return pcd, [np.stack(cam2world, axis=0) for cam2world in cam2worlds], lidar_on_image_mask_ls

def visualize(geometries,
              name: str = "Main",
              width: int = 960,
              height: int = 540,
              ):
    point_size = 5
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.light_on = False
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()
    return

def visualize_groups(
    geom_list_list,
    window_name: str = "Viewer",
    width: int = 960,
    height: int = 540,
    point_size: int = 3.5,
):
    if not geom_list_list:
        print("[visualize_groups] geom_list_list is empty, nothing to show.")
        return

    # 过滤空组
    geom_list_list = [gl for gl in geom_list_list if gl]
    if not geom_list_list:
        print("[visualize_groups] all geometry groups are empty.")
        return

    num_groups = len(geom_list_list)
    current_idx = 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=width, height=height)

    opt = vis.get_render_option()
    opt.light_on = False
    opt.point_size = point_size

    view_ctl = vis.get_view_control()

    def show_group(idx, restore_view=False, prev_params=None):
        vis.clear_geometries()
        for g in geom_list_list[idx]:
            vis.add_geometry(g)
        vis.update_renderer()

        if restore_view and prev_params is not None:
            view_ctl.convert_from_pinhole_camera_parameters(prev_params)
        else:
            vis.reset_view_point(True)

        print(f"[visualize_groups] Showing group {idx + 1}/{num_groups}")

    show_group(current_idx)

    def next_group(vis_):
        nonlocal current_idx

        prev_params = view_ctl.convert_to_pinhole_camera_parameters()

        current_idx = (current_idx + 1) % num_groups

        show_group(current_idx, restore_view=True, prev_params=prev_params)

        return False  

    vis.register_key_callback(ord("N"), next_group)
    vis.register_key_callback(ord("n"), next_group)

    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()

def set_axes_equal(ax):
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
    
def chamfer_distance_RMSE(ref, est, max_error):
    ref = np.asarray(ref.points)
    est = np.asarray(est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    chamfer_dist = 0.5 * np.mean(dist1) + 0.5 * np.mean(dist2)

    return chamfer_dist, np.mean(dist1), np.mean(dist2)

def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]
    if N < 3:
        raise ValueError("at least 3 points are required to compute Umeyama alignment")

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]; Y = Y[mask]
    N = X.shape[0]
    if N < 3:
        raise ValueError("at least 3 valid correspondences are required to estimate similarity transform")

    mu_X = X.mean(axis=0); mu_Y = Y.mean(axis=0)
    Xc = X - mu_X; Yc = Y - mu_Y
    Sigma = (Yc.T @ Xc) / N  # 3x3

    U, D, Vt = np.linalg.svd(Sigma)  # D: (3,)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2,2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_X = (Xc**2).sum() / N
        s = float(np.sum(D * np.diag(S)) / var_X)  
    else:
        s = 1.0

    t = mu_Y - s * (R @ mu_X)
    return s, R, t




def _proj_to_SO3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def _estimate_global_rotation_SO3(R_pred_list: np.ndarray, R_gt_list: np.ndarray, weights: np.ndarray=None) -> np.ndarray:
    """
    Closed-form solution: min_R sum || R R_pred - R_gt ||_F^2 (chordal L2)
    """
    N = R_pred_list.shape[0]
    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    M = np.zeros((3,3), dtype=np.float64)
    for i in range(N):
        M += weights[i] * (R_gt_list[i] @ R_pred_list[i].T)
    return _proj_to_SO3(M)

def _rotation_angle_deg(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))

def _compute_metrics(gt_T: np.ndarray, pred_T: np.ndarray, rpe_delta: int = 1) -> dict:
    N = gt_T.shape[0]
    e_t = pred_T[:, :3, 3] - gt_T[:, :3, 3]
    ATE_RMSE = float(np.sqrt(np.mean(np.sum(e_t**2, axis=1))))
    if N - rpe_delta <= 0:
        return {"ATE_RMSE": ATE_RMSE, "RTE_RMSE": float('nan'), "RRE_RMSE": float('nan')}
    rtes, rres = [], []
    for i in range(N - rpe_delta):
        Tgi = np.linalg.inv(gt_T[i]) @ gt_T[i + rpe_delta]
        Tpi = np.linalg.inv(pred_T[i]) @ pred_T[i + rpe_delta]
        Terr = np.linalg.inv(Tgi) @ Tpi
        rtes.append(np.linalg.norm(Terr[:3, 3]))
        rres.append(_rotation_angle_deg(Terr[:3, :3]))
    RTE_RMSE = float(np.sqrt(np.mean(np.array(rtes)**2))) if rtes else float('nan')
    RRE_RMSE = float(np.sqrt(np.mean(np.array(rres)**2))) if rres else float('nan')
    return {"ATE_RMSE": ATE_RMSE, "RTE_RMSE": RTE_RMSE, "RRE_RMSE": RRE_RMSE}

def align_pred_cam2world_to_gt(gt_cam2world: np.ndarray,
                               pred_cam2world: np.ndarray,
                               estimate_scale: bool = True,
                               return_metrics: bool = True,
                               visualize: bool = True,
                               rpe_delta: int = 1,
                               correct_heading: bool = True):
    """
    Steps:
      1) Perform Umeyama alignment on the camera center trajectories to obtain s_u, R_u, t_u, and apply to the entire pred trajectory;
      2) Based on all frame rotations, solve for a global rotation R_g and perform centroid translation t_g;
      3) Compose the final alignment: R = R_g @ R_u, t = R_g @ t_u + t_g, s = s_u.
    Returns (maintaining compatibility with unpacking order):
        R: (3,3)  —— Final global rotation (can be directly applied to point clouds)
        t: (3,)   —— Final global translation (can be directly applied to point clouds)
        s: float  —— Scale (from Umeyama; should also be applied to point clouds)
        pred_cam2world_aligned: (N,4,4) —— Camera poses after applying the full (Sim(3)+R_g,t_g) alignment
    If return_metrics=True, also returns:
        metrics: dict containing 'ATE_RMSE','RTE_RMSE','RRE_RMSE'
    """
    # ---- Input checks ----
    if not (gt_cam2world.ndim == 3 and gt_cam2world.shape[1:] == (4, 4)):
        raise ValueError("gt_cam2world shape should be (N,4,4)")
    assert pred_cam2world.ndim == 3 and pred_cam2world.shape[1:] == (4, 4)
    if gt_cam2world.shape[0] != pred_cam2world.shape[0]:
        raise ValueError("gt and pred have different number of frames")
    N = gt_cam2world.shape[0]

    # ---- Step 1: Umeyama (on camera centers) to estimate Sim(3) ----
    Cw_gt   = gt_cam2world[:, :3, 3]    # (N,3)
    gt_traj_lenth = np.sum(np.linalg.norm(np.diff(Cw_gt, axis=0), axis=1))
    print(f"GT trajectory length: {gt_traj_lenth:.3f} meters")
    Cw_pred = pred_cam2world[:, :3, 3]  # (N,3)

    mu_pred = np.mean(Cw_pred, axis=0)
    mu_gt   = np.mean(Cw_gt,   axis=0)
    X = Cw_pred - mu_pred
    Y = Cw_gt   - mu_gt

    Sigma = (Y.T @ X) / N  # 3x3
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R_u = U @ S @ Vt

    if estimate_scale:
        var_X = np.mean(np.sum(X**2, axis=1))
        s_u = float(np.sum(D * np.diag(S)) / (var_X + 1e-12))
    else:
        s_u = 1.0

    t_u = mu_gt - s_u * (R_u @ mu_pred)

    # Apply Umeyama alignment to the entire trajectory
    pred_after_sim3 = np.empty_like(pred_cam2world)
    for i in range(N):
        R_pred = pred_cam2world[i, :3, :3]
        t_pred = pred_cam2world[i, :3, 3]
        R_new  = R_u @ R_pred
        t_new  = s_u * (R_u @ t_pred) + t_u
        T = np.eye(4, dtype=pred_cam2world.dtype)
        T[:3, :3] = R_new
        T[:3, 3]  = t_new
        pred_after_sim3[i] = T

    metrics = _compute_metrics(gt_cam2world, pred_after_sim3, rpe_delta=rpe_delta) if return_metrics else None

    if not correct_heading:
        return R_u, t_u, s_u, pred_after_sim3 if not return_metrics else (R_u, t_u, s_u, pred_after_sim3, metrics)

    # ---- Step 2: Global rotation R_g + centroid translation t_g (single, not per-frame)----
    R_pred_list = pred_after_sim3[:, :3, :3]
    R_gt_list   = gt_cam2world[:, :3, :3]
    R_g = _estimate_global_rotation_SO3(R_pred_list, R_gt_list, weights=None)

    pred_centroid = pred_after_sim3[:, :3, 3].mean(axis=0)
    gt_centroid   = gt_cam2world[:, :3, 3].mean(axis=0)
    t_g = gt_centroid - R_g @ pred_centroid

    # 应用 (R_g, t_g)
    pred_aligned = np.empty_like(pred_after_sim3)
    for i in range(N):
        R_old = pred_after_sim3[i, :3, :3]
        t_old = pred_after_sim3[i, :3, 3]
        R_new = R_g @ R_old
        t_new = R_g @ t_old + t_g
        T = np.eye(4, dtype=pred_after_sim3.dtype)
        T[:3, :3] = R_new
        T[:3, 3]  = t_new
        pred_aligned[i] = T

    # ---- Compose final global (R, t, s) that can be directly applied to point clouds ----
    # Point clouds from pred frame to gt frame: X_aligned = s * (R @ X) + t
    R = R_g @ R_u
    t = (R_g @ t_u) + t_g
    s = float(s_u)


    if visualize:
        # plt.figure()
        # plt.plot(gt_cam2world[:,0,3],  gt_cam2world[:,1,3],  label="GT", linewidth=2)
        # plt.plot(pred_cam2world[:,0,3], pred_cam2world[:,1,3], label="Pred (raw)", alpha=0.6)
        # plt.plot(pred_after_sim3[:,0,3], pred_after_sim3[:,1,3], label="After Umeyama")
        # plt.plot(pred_aligned[:,0,3],    pred_aligned[:,1,3],    label="After Umeyama + Rg", linewidth=2)
        # plt.axis('equal'); plt.legend(); plt.title("Top-Down (x-y) Trajectories"); plt.xlabel("x"); plt.ylabel("y")
        # plt.tight_layout(); plt.show()

        # 3D View
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(gt_cam2world[:,0,3],  gt_cam2world[:,1,3],  gt_cam2world[:,2,3], label="GT", c="blue", linewidth=4)
        ax1.plot(pred_after_sim3[:,0,3], pred_after_sim3[:,1,3], pred_after_sim3[:,2,3], label="After Umeyama", c="orange", linestyle=":", linewidth=4)
        # ax1.plot(pred_aligned[:, 0, 3], pred_aligned[:, 1, 3], pred_aligned[:, 2, 3], label="After Umeyama + Rg", c="red", linestyle="--", linewidth=4)
        ax1.set_title("3D Trajectory")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
        ax1.legend(); ax1.grid(True)
        set_axes_equal(ax1)

        plt.tight_layout()
        plt.show()
        # visualize_trajectories_o3d(Cw_gt, pred_aligned, stride=1)

    if return_metrics:
        return R, t, s, pred_aligned, metrics
    else:
        return R, t, s, pred_aligned

def apply_sim3_to_pcd(pcd, R, t, s):
    """p' = s * R * p + t"""
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts_t = (s * (R @ pts.T) + t[:,None]).T
    pcd_new = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_t))
    if pcd.has_colors():
        pcd_new.colors = pcd.colors
    if pcd.has_normals():
        pcd_new.normals = pcd.normals
    return pcd_new

def apply_sim3_to_cam2world(cam2worlds, R, t, s):
    """
    Apply a global Sim(3) transform to a batch of cam2world matrices.

    Sim(3): p' = s * R * p + t

    Given cam2world T_cw: X_w = T_cw @ X_c,
    the new pose under Sim(3) in world coordinates is:
        X'_w = S @ X_w
    so:
        T'_cw = S @ T_cw
    where S = [[sR, t],
               [ 0, 1]]

    Args:
        cam2worlds: (N, 4, 4) array-like, original cam2world matrices.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        s: float, uniform scale.

    Returns:
        (N, 4, 4) numpy array: transformed cam2worlds.
    """
    cam2worlds = np.asarray(cam2worlds, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3,)

    assert cam2worlds.ndim == 3 and cam2worlds.shape[1:] == (4, 4), "cam2worlds must be (N,4,4)"
    assert R.shape == (3, 3), "R must be (3,3)"
    assert t.shape == (3,), "t must be (3,)"

    S = np.eye(4, dtype=np.float64)
    S[:3, :3] = s * R
    S[:3, 3] = t

    # Broadcast left-multiply: T'_cw = S @ T_cw
    cam2worlds_new = S[None, :, :] @ cam2worlds
    return cam2worlds_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR GT vs Pred dense point cloud evaluation")
    parser.add_argument("--GT", default="Data/nuscenes/scene-0003/")
    parser.add_argument("--pred", default="Data/nuscenes/scene-0012/xx")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_for_drawing", action="store_true")

    # evaluation options
    parser.add_argument("--eval_traj", action="store_true")
    parser.add_argument("--vis_eval_traj", action="store_true")
    parser.add_argument("--eval_pcd", action="store_true")
    parser.add_argument("--vis_eval_pcd", action="store_true")
    parser.add_argument("--max_error", type=float, default=10, help="Truncation for distances (meters) in metrics")

    args = parser.parse_args()

    if args.vis:
        # visualize(vis_GT(args.GT, args.pred, dataset='nuscenes' if 'nuscenes' in args.GT else 'waymo'), name="GT")
        visualize(vis_pred(args.GT, args.pred, dataset='nuscenes' if 'nuscenes' in args.pred else 'waymo'), name="Pred")
        # visualize_groups([vis_pred(args.GT, args.pred, dataset='nuscenes' if 'nuscenes' in args.pred else 'waymo'), 
        #                   vis_pred(args.GT, args.pred.replace("tps", "sim3"), dataset='nuscenes' if 'nuscenes' in args.pred else 'waymo'),
        #                   vis_pred(args.GT, args.pred.replace("tps", "sl4"), dataset='nuscenes' if 'nuscenes' in args.pred else 'waymo'),
        #                 ])
    
    if args.eval_traj or args.eval_pcd:
        gt_pcd, gt_cam2world, lidar_on_image_mask_ls = load_GT(args.GT, pred_path=args.pred, load_pcd=args.eval_pcd)
        pred_pcd, pred_cam2world = load_pred(args.pred, load_pcd=args.eval_pcd)
        gt_cam2world_ref, pred_cam2world_ref = gt_cam2world[0], pred_cam2world[0]
        ###############################
        metrics = {}
        if args.eval_traj or args.eval_pcd:
            R_a, t_a, s, pred_cam2world_ref, metrics = align_pred_cam2world_to_gt(gt_cam2world_ref, pred_cam2world_ref, visualize=args.vis_eval_traj)
            metrics = {
                "trajectory": metrics
            }

        
        if args.eval_pcd:
            pred_pcd = apply_sim3_to_pcd(pred_pcd, R_a, t_a, s)
            if args.vis_eval_pcd:
                pred_cam2world = [apply_sim3_to_cam2world(p, R_a, t_a, s) for p in pred_cam2world]
                # ipdb.set_trace()
                # assert np.allclose(pred_cam2world[0], pred_cam2world_ref)
                pred_geoms = [pred_pcd]
                cam_list = cam_list_nuscenes if 'nuscenes' in args.GT else cam_list_waymo
                shape_list = [cv2.imread(join(args.GT, f"image/{cam}/{0:0>3d}.jpg")).shape for cam in cam_list]
                intrinsic3x3_dict = [np.loadtxt(join(args.GT, f"intrinsic/{cam}.txt")) for cam in cam_list]
                color_list = list(camera_color_map_waymo.values()) if 'waymo' in args.GT else list(camera_color_map_nuscenes.values())
                for cam in range(len(pred_cam2world)):
                    for i in range(pred_cam2world[cam].shape[0]):
                        pred_geoms.extend(create_camera_frustum(pred_cam2world[cam][i], intrinsic3x3_dict[cam], (shape_list[cam][0], shape_list[cam][1]), color_list[cam], 1.5))
                visualize(pred_geoms, comparison_geometries=vis_GT(args.GT, args.pred, dataset='nuscenes' if 'nuscenes' in args.GT else 'waymo'))

            
            # 4) Metrics (Pred->GT accuracy, GT->Pred completeness, symmetric Chamfer)
            chamfer, acc, comp = chamfer_distance_RMSE(
                gt_pcd, pred_pcd, max_error=args.max_error
            )
            metrics.update({
                "pcd_after_icp": {
                    "acc": acc,
                    "comp": comp,
                    "chamfer": chamfer,

                }
            })
            print(metrics)

            if args.vis_eval_pcd:
                visualize([gt_pcd], comparison_geometries=[pred_pcd])


        if args.eval_traj or args.eval_pcd:
            json.dump(metrics, open(join(args.pred, "eval_metrics.json"), "w"), indent=4)

        