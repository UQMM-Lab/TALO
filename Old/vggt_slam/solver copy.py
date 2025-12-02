import os
import time
from typing import Dict, List
import ipdb
import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer
from vggt_slam.h_solve import apply_homography, apply_homography_batch

from rig_solver import *
from interframe_solver import *
from third_party_utils import load_images_as_tensor, robust_weighted_estimate_sim3, apply_sim3, recover_focal_shift
from pi3.utils.geometry import homogenize_points, depth_edge
import utils3d

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"

    # Normalize confidence to [0, 1]
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    
    # Map to colors using matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]  # Drop alpha channel

    # Assign to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # 只控制 Frustum 的可见性
        self.gui_show_frustums = self.server.gui.add_checkbox(
            "Show Camera Frustums",
            initial_value=True,
        )
        self.gui_show_frustums.on_update(self._on_update_show_frustums)

        # 只存 frustum 句柄
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        extrinsics: (S, 3, 4)  相机 cam->world
        images_:    (S, 3, H, W)
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frustums:
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # 不再创建坐标系 frame，frustum 直接挂在根节点
            frustum_name = f"submap_{submap_id}/cam_{img_id}"

            # 准备图像
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # 估算视场角（如有内参，建议用 fy_pixels = K[1,1]）
            fy_pixels = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy_pixels)

            # 直接把位姿赋给 frustum（取代原来的父子关系）
            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                wxyz=T_world_camera.rotation().wxyz,      # 位置与朝向
                position=T_world_camera.translation(),
                fov=fov,
                aspect=w / h,
                scale=0.005,
                image=img,
                line_width=3.0,
                color=self.random_colors[submap_id % len(self.random_colors)],
            )
            frustum.visible = self.gui_show_frustums.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frustums(self, _) -> None:
        visible = self.gui_show_frustums.value
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible


class Solver:
    def __init__(self,
        model,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        interframe_solver_choice: str = "pose",
        gradio_mode: bool = False,
        port = 8080,
        overlap_size = 1,
        submap_size = 1,
        cam_names = [],
        ref_cam_id = -1,
        ref_cam_name = '',
        vis_tps=False,
        allow_nonvalid=False,
        voxel_size=0.05,
        max_project_error=0.1,
        filter_method="mean",
        filter_k=32,
        robust_mean_method=None,
        backward_control_points=False,
        ):
        self.model = model
        if model == "VGGT":
            self.load_and_preprocess_images = load_and_preprocess_images
        elif model == "Pi3":
            self.load_and_preprocess_images = load_images_as_tensor
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if visualize_global_map:
            if self.gradio_mode:
                self.viewer = TrimeshViewer()
            else:
                self.viewer = Viewer(port=port)

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.interframe_solver_choice = interframe_solver_choice
        if interframe_solver_choice.startswith("pose") or interframe_solver_choice.startswith("sim3"):
            from vggt_slam.graph_se3 import PoseGraph
        elif interframe_solver_choice.startswith("sl4"):
            from vggt_slam.graph import PoseGraph
        else:
            raise
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None
        self.cam_names = cam_names
        self.cam_num = len(cam_names)
        self.overlap_num = overlap_size * self.cam_num
        self.submap_num = submap_size * self.cam_num
        self.ref_cam_id = ref_cam_id
        self.ref_cam_name = ref_cam_name
        self.control_points = {}
        self.submap_control_points = {}
        self.vis_tps = vis_tps
        self.allow_nonvalid = allow_nonvalid
        self.voxel_size = voxel_size
        self.max_project_error = max_project_error
        self.filter_method = filter_method
        self.filter_k = filter_k
        self.robust_mean_method = robust_mean_method
        self.backward_cps = backward_control_points

        print("Starting viser server...")

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_"+name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_filtered_points_in_world_frame()
        points_colors = submap.get_filtered_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world()  # (S, 3, 4)
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self, vis_point=True):
        for submap in self.map.get_submaps():
            self.set_submap_poses(submap)
            if vis_point:
                self.set_submap_point_cloud(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)
        time.sleep(2)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        
        

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)

        new_pcd_num = self.current_working_submap.get_id()
        self.submap_control_points[self.current_working_submap.get_id()] = []
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[self.submap_num:self.submap_num+self.overlap_num]
            self.prior_conf = conf[self.submap_num:self.submap_num+self.overlap_num]

            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
            
            self.current_working_submap.set_reference_homography(H_w_submap)
            self.current_working_submap.add_all_poses(cam_to_world, cam_to_world.copy())
            self.current_working_submap.add_all_points(world_points, colors, intrinsics_cam)
            if self.model == "VGGT":
                self.current_working_submap.add_all_conf(conf, np.percentile(conf, self.init_conf_threshold))
            elif self.model == "Pi3":
                self.current_working_submap.add_all_conf(conf, 0.4)
            self.map.add_submap(self.current_working_submap)

        else:
            # visualize_correspondences(world_points[:self.overlap_num].reshape(-1, 3),
            #                           world_points[self.overlap_num:self.overlap_num+self.submap_num].reshape(-1, 3))
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)
            if self.model == "VGGT":
                self.current_working_submap.add_all_conf(conf[self.overlap_num:self.overlap_num+self.submap_num], np.percentile(conf, self.init_conf_threshold))
            elif self.model == "Pi3":
                self.current_working_submap.add_all_conf(conf[self.overlap_num:self.overlap_num+self.submap_num], 0.4)

            current_pts = world_points[:self.overlap_num]
        
            # TODO conf should be using the threshold in its own submap
            good_mask = (self.prior_conf >= prior_submap.get_conf_threshold()) & (conf[:self.overlap_num] >= self.current_working_submap.get_conf_threshold())
            # good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[:self.overlap_num].reshape(-1) > prior_submap.get_conf_threshold())

            if self.interframe_solver_choice.startswith("pose"):
                H_relative = prior_submap.poses[:self.overlap_num+self.submap_num][-self.overlap_num:][self.ref_cam_id]
                if self.interframe_solver_choice.startswith("pose+scale"):

                    R = H_relative[:3, :3].astype(np.float64)
                    t = H_relative[:3,  3].astype(np.float64)

                    X = current_pts[good_mask].astype(np.float64)  # x_i
                    Y = self.prior_pcd[good_mask].astype(np.float64)   # y_i

                    Xp = (R @ X.T).T            # x_i' = R x_i
                    Yp = Y - t[None, :]         # y_i' = y_i - t

                    eps = 1e-12

                    denom = np.sum(Xp * Xp) + eps
                    num   = np.sum(Xp * Yp)
                    s = num / denom

                    # denom_i = np.sum(Xp * Xp, axis=1) + eps
                    # num_i   = np.sum(Xp * Yp, axis=1)
                    # s_i = num_i / denom_i
                    # # 鲁棒聚合：中位数；也可换 Huber 加权平均
                    # s = np.median(s_i[np.isfinite(s_i)])

                    s = np.clip(s, 0.7, 1.3)
                    print(colored("scale factor", 'green'), s)
                    cam_to_world[:, 0:3, 3] *= s
                    world_points *= s
                    current_pts = world_points[:self.overlap_num]

                # weights = np.minimum(self.prior_conf[good_mask], conf[:self.overlap_num][good_mask])
                # H_relative = weighted_umeyama_sim3(current_pts[good_mask], self.prior_pcd[good_mask], weights=weights)
            elif self.interframe_solver_choice == "sim3":
                s, R, t = robust_weighted_estimate_sim3(
                    src=current_pts[good_mask], 
                    tgt=self.prior_pcd[good_mask],
                    init_weights=np.sqrt(self.prior_conf[good_mask] * conf[:self.overlap_num][good_mask]),
                )
                pcd1 = apply_sim3(current_pts[good_mask], s, R, t)
                H_relative = np.eye(4)
                H_relative[:3, :3] = R
                H_relative[:3,  3] = t
                world_points *= s
                current_pts = world_points[:self.overlap_num]
                pcd2 = apply_homography(H_relative, current_pts[good_mask])
                if not np.allclose(pcd1, pcd2, atol=1e-4):
                    ipdb.set_trace()

                cam_to_world[:, 0:3, 3] *= s
                if not self.use_point_map:
                    depth_map *= s
            elif self.interframe_solver_choice == "sl4-onlyref":
                H_relative = ransac_projective(current_pts[self.ref_cam_id][good_mask[self.ref_cam_id]], self.prior_pcd[self.ref_cam_id][good_mask[self.ref_cam_id]])
            elif self.interframe_solver_choice == "sl4":
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            #########################################################
            #########################################################
            #########################################################


            H_w_submap = prior_submap.get_reference_homography() @ H_relative


            self.prior_pcd = world_points[:self.overlap_num+self.submap_num][-self.overlap_num:]
            self.prior_conf = conf[:self.overlap_num+self.submap_num][-self.overlap_num:]

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)
            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)


            print("added between factor", prior_pcd_num, new_pcd_num)

            self.current_working_submap.set_reference_homography(H_w_submap)
            self.current_working_submap.add_all_poses(cam_to_world[:self.overlap_num+self.submap_num], cam_to_world[self.overlap_num:self.overlap_num+self.submap_num])
            self.current_working_submap.add_all_points(world_points[self.overlap_num:self.overlap_num+self.submap_num], 
                                                       colors[self.overlap_num:self.overlap_num+self.submap_num], 
                                                       intrinsics_cam[self.overlap_num:self.overlap_num+self.submap_num])
            # self.current_working_submap.add_all_depths(depth_map)
            
        
            self.map.add_submap(self.current_working_submap)
            # self.optimize_inner_poses()
            if self.interframe_solver_choice.endswith("tps"):
                H, W = self.current_working_submap.get_HW()

                old_control_points_idx, old_control_points_3d, old_control_points_2d, old_control_points_color = self.forward_control_points(prior_submap.get_id(), current_pts)

                # —— 控制点选择 ——
                old_control_points_keep_idx, new_control_points_idx = select_control_points_np(
                    current_pts[good_mask],  # 选择阶段用 float32 即可
                    max_control_points=10000,
                    method="voxel",         # "random" | "voxel" | "octree"
                    kwargs={"voxel_size":self.voxel_size, "min_threshold":1},  # 比如 {"voxel_size":0.1} 或 {"max_depth":7,"leaf_points":256}
                    # vis=True,
                    given_control_points=old_control_points_3d,
                    keep_all_given=self.allow_nonvalid
                )
                print(colored(f"Found {len(old_control_points_keep_idx)} old control points, {len(new_control_points_idx)} new control points", "cyan"))
                # extract attributes of new control points
                orig_idx = np.flatnonzero(good_mask)[new_control_points_idx]
                xs, ys, zs = np.unravel_index(orig_idx, self.prior_conf.shape)
                new_control_points_2d_overlap = np.stack([xs, zs, ys], axis=1)
                new_control_points_color_overlap = colors[:self.overlap_num][good_mask][new_control_points_idx]
                new_control_points_3d = current_pts[good_mask][new_control_points_idx]
                
                # prepare for projection
                cam_to_world_ = self.current_working_submap.poses[-self.overlap_num:]
                intrinsics_cam_ = intrinsics_cam[:self.overlap_num+self.submap_num][-self.overlap_num:]
                current_pts_ = world_points[:self.overlap_num+self.submap_num][-self.overlap_num:]
                valid_mask = conf[:self.overlap_num+self.submap_num][-self.overlap_num:]>=self.current_working_submap.get_conf_threshold()
                colors_ = colors[:self.overlap_num+self.submap_num][-self.overlap_num:]
                # project old control points 
                if old_control_points_keep_idx.shape[0] > 0:
                    old_control_points_3d = old_control_points_3d[old_control_points_keep_idx]
                    old_control_points_color = old_control_points_color[old_control_points_keep_idx]
                    old_control_points_2d = old_control_points_2d[old_control_points_keep_idx]
                    old_control_points_2d, old_control_points_color = project(
                        points_world=old_control_points_3d,
                        points_cam=old_control_points_2d[:, 0],
                        points_color=old_control_points_color,

                        points_world_check=current_pts_, 
                        points_color_check=colors_,
                        cam2world=cam_to_world_, intrinsics=intrinsics_cam_, 
                        H=H, W=W, 
                        valid_mask=valid_mask if not self.allow_nonvalid else None, 
                        max_error_3d=self.max_project_error
                    )
                    if self.vis_tps:
                        check_3d = np.array([current_pts_[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
                        check_3d_2 = np.array([old_control_points_3d[i] for i in range(len(old_control_points_2d)) if old_control_points_2d[i] is not None])
                        visualize_correspondences(check_3d, check_3d_2, window_name="Old Innerframe, not allow errors", background=(current_pts[good_mask], colors[:self.overlap_num][good_mask]))
                    for i, point_id in enumerate(old_control_points_keep_idx):
                        self.control_points[old_control_points_idx[point_id]][self.current_working_submap.get_id()] = {
                            "3d": old_control_points_3d[i],
                            '2d': old_control_points_2d[i],
                            'color': old_control_points_color[i]
                        }
                    self.submap_control_points[self.current_working_submap.get_id()] += [old_control_points_idx[i] for i in old_control_points_keep_idx]

                # project new control points
                new_control_points_2d, new_control_points_color = project(
                    points_world=new_control_points_3d, 
                    points_cam=new_control_points_2d_overlap[:,0], 
                    points_color=new_control_points_color_overlap,

                    points_world_check=current_pts_, 
                    points_color_check=colors_, 
                    cam2world=cam_to_world_, intrinsics=intrinsics_cam_, 
                    H=H, W=W, 
                    valid_mask=valid_mask if not self.allow_nonvalid else None, 
                    max_error_3d=self.max_project_error
                )
                if self.vis_tps:
                    check_3d = np.array([current_pts_[p[0], p[2], p[1]] for p in new_control_points_2d if p is not None])
                    check_3d_2 = np.array([new_control_points_3d[i] for i in range(len(new_control_points_2d)) if new_control_points_2d[i] is not None])
                    visualize_correspondences(check_3d, check_3d_2, window_name="New Innerframe, not allow errors", background=(world_points[conf>=self.current_working_submap.get_conf_threshold()], colors[conf>=self.current_working_submap.get_conf_threshold()]))
                start_idx = len(self.control_points)
                for point_id in range(new_control_points_3d.shape[0]):
                    self.control_points[start_idx + point_id] = {
                            self.current_working_submap.get_id(): {
                                "3d": new_control_points_3d[point_id],
                                "2d": new_control_points_2d[point_id],
                                'color': new_control_points_color[point_id]
                            }
                    }
                self.submap_control_points[self.current_working_submap.get_id()] += [start_idx + i for i in range(new_control_points_3d.shape[0])]


                if self.current_working_submap.get_id() == 1 or self.backward_cps:
                    # backward new control points to all previous submaps
                    
                    
                    new_control_points_idx = [start_idx + i for i in range(new_control_points_3d.shape[0])]

                    submap_id = -1
                    print(self.map.submaps.keys())
                    new_control_points_2d = new_control_points_2d_overlap
                    new_control_points_color = new_control_points_color_overlap
                    for submap_id in list(self.map.submaps.keys())[::-1][1:]:
                        try:
                            assert len(new_control_points_2d) == len(new_control_points_3d) == len(new_control_points_idx) == len(new_control_points_color)
                        except:
                            ipdb.set_trace()
                        new_control_points_2d, new_control_points_3d, new_control_points_idx, new_control_points_color = self.backward_control_points(
                            old_control_points_2d=new_control_points_2d,
                            old_control_points_3d=new_control_points_3d,
                            old_control_points_idx=new_control_points_idx,
                            old_control_points_color=new_control_points_color,
                            tgt_submap_id=submap_id,
                        )
                        if len(new_control_points_idx) == 0:
                            print(colored(f"stop backward at submap {submap_id} for {self.current_working_submap.get_id()} due to no control points", "yellow"))
                            break
            
        
        

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.submap_num + self.overlap_num + index

            if self.interframe_solver_choice.startswith("pose") or self.interframe_solver_choice.startswith("sim3"):
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            elif self.interframe_solver_choice.startswith("sl4"):
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id)
            # print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

            # print("relative_pose factor added", relative_pose)

            # Visualize query and detected frames
            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(loop.detected_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(loop.query_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()

            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()
        
        



        
        

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, frame_ids=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = self.load_and_preprocess_images(image_names).to(device)
        # print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Check for loop closures
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        # new_submap.add_all_frames(images)
        new_submap.add_all_frames(images)
        if frame_ids is None:
            new_submap.set_frame_ids(image_names)
        else:
            new_submap.frame_ids = frame_ids
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap, self.cam_num, self.ref_cam_id))

        # TODO implement this
        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        num_loop_frames = len(retrieved_frames)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            # new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        if self.model == "Pi3":
            images = images[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        

        if self.model == "VGGT":
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            predictions["detected_loops"] = detected_loops

            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
            return predictions
        elif self.model == "Pi3":
            results = {}
            cam2world = torch.inverse(predictions['camera_poses'][0][0]) @ predictions['camera_poses'][0] # (S, 4, 4)
            world2cam = torch.inverse(cam2world)
            results['extrinsic'] = world2cam.cpu().numpy() # (S, 4, 4)

            results['world_points'] = torch.einsum('nij, nhwj -> nhwi', cam2world, homogenize_points(predictions['local_points'][0]))[..., :3].cpu().numpy() # (S, H, W, 3)
            results['world_points_conf'] = torch.sigmoid(predictions['conf'][0,...,0]).cpu().numpy() # (S, H, W)
            results['images'] = images[0].cpu().numpy() # (S, 3, H, W)
            results["detected_loops"] = detected_loops
            results['intrinsic'] = [None]*results['extrinsic'].shape[0]


            points = predictions["local_points"]
            masks = torch.sigmoid(predictions['conf'][..., 0]) > 0.1
            non_edge = ~depth_edge(predictions['local_points'][..., 2], rtol=0.03)
            masks = torch.logical_and(masks, non_edge)[0]

            original_height, original_width = points.shape[-3:-1]
            aspect_ratio = original_width / original_height
            # use recover_focal_shift function from MoGe
            focal, shift = recover_focal_shift(points, masks)
            fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
            zero_point_five = torch.full_like(fx, 0.5)
            intrinsics = torch.stack([
                fx, zeros, zero_point_five, 
                zeros, fy, zero_point_five, 
                zeros, zeros, ones
            ], dim=-1).reshape(-1, 3, 3)
            results['intrinsic'] = intrinsics.cpu().numpy()

            return results



    def forward_control_points(self, src_sid, points_3d):
        old_control_points_2d_dict = {}
        old_control_points_3d_check = []
        old_control_points_color = []
        for point_id in self.submap_control_points.get(src_sid, []):
            point2d = self.control_points[point_id][src_sid]["2d"]
            if point2d is not None:
                old_control_points_2d_dict[point_id] = point2d
                old_control_points_3d_check.append(self.control_points[point_id][src_sid]["3d"])
                old_control_points_color.append(self.control_points[point_id][src_sid]["color"])
        if len(old_control_points_3d_check) > 0:
            old_control_points_3d_check = np.stack(old_control_points_3d_check, axis=0)
            old_control_points_2d = list(old_control_points_2d_dict.values())
            old_control_points_idx = list(old_control_points_2d_dict.keys())
            old_control_points_color = np.array(old_control_points_color)
            old_control_points_3d = np.array([points_3d[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
            if self.vis_tps:
                visualize_correspondences(old_control_points_3d, old_control_points_3d_check, window_name="Forward, Interframe, allow errors")
        else:
            return None, None, None, None
        return old_control_points_idx, old_control_points_3d, np.array(old_control_points_2d), old_control_points_color

    def backward_control_points(self, old_control_points_2d, old_control_points_3d, old_control_points_idx, old_control_points_color, tgt_submap_id):
        tgt_submap = self.map.get_submap(tgt_submap_id)
        cam_to_world = tgt_submap.get_all_poses_world()  # (S, 3, 4)
        intrinsics_cam = tgt_submap.get_all_intrinsics()  # (S, 3, 3)
        valid_mask = tgt_submap.get_all_conf() >= tgt_submap.get_conf_threshold()  # (S, H, W)
        pts_3d = tgt_submap.get_all_points()[:self.overlap_num+self.submap_num][-self.overlap_num:]  
        colors = tgt_submap.get_all_colors()

        if not self.allow_nonvalid and tgt_submap_id < self.current_working_submap.get_id()-1:
            old_control_points_2d_tp = []
            old_control_points_idx_tp = []
            old_control_points_color_tp = []
            old_control_points_3d_tp = []
            valid_mask_tp = valid_mask[:self.overlap_num+self.submap_num][-self.overlap_num:]
            for i, p in enumerate(old_control_points_2d):
                if valid_mask_tp[p[0], p[2], p[1]]:
                    old_control_points_2d_tp.append(p)
                    old_control_points_idx_tp.append(old_control_points_idx[i])
                    old_control_points_color_tp.append(old_control_points_color[i])
                    old_control_points_3d_tp.append(old_control_points_3d[i])
            # if tgt_submap_id == self.current_working_submap.get_id()-1:
            #     assert len(old_control_points_2d) == len(old_control_points_2d_tp)
            old_control_points_2d = np.array(old_control_points_2d_tp)
            old_control_points_idx = old_control_points_idx_tp
            old_control_points_color = old_control_points_color_tp
            old_control_points_3d = np.array(old_control_points_3d_tp)

        control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
        if self.vis_tps:
            # old_control_points_3d = np.array([old_control_points_3d[i] for i in range(len(old_control_points_2d)) if old_control_points_2d[i] is not None])
            visualize_correspondences(control_points_3d, old_control_points_3d, window_name=f"Backward, Interframe, allow errors, {len(control_points_3d)}/{len(old_control_points_3d)} points")

        for i, point_id in enumerate(old_control_points_idx):
            self.control_points[point_id][tgt_submap_id] = {
                "3d": control_points_3d[i],
                # '2d': control_points_2d[i]
            }
        self.submap_control_points[tgt_submap_id].extend(old_control_points_idx)
        if tgt_submap_id == 0 or len(control_points_3d) == 0:
            return [], [], [], []

        H, W = tgt_submap.get_HW()
            
        control_points_2d, control_points_color = project(
            points_world=control_points_3d,
            points_cam=old_control_points_2d[:, 0],
            points_color=old_control_points_color,
            
            points_world_check=tgt_submap.get_all_points()[:self.overlap_num], 
            points_color_check=colors[:self.overlap_num],
            cam2world=cam_to_world[:self.overlap_num], intrinsics=intrinsics_cam[:self.overlap_num], 
            H=H, W=W, 
            valid_mask=valid_mask[:self.overlap_num] if not self.allow_nonvalid else None, 
            max_error_3d=self.max_project_error
        )
        
        control_points_3d = np.array([control_points_3d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None])
        control_points_idx = [old_control_points_idx[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None]
        control_points_color = [control_points_color[i] for i in range(len(control_points_color)) if control_points_2d[i] is not None]
        control_points_2d = np.array([control_points_2d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None])
        if self.vis_tps:
            pts_3d = tgt_submap.get_all_points()[:self.overlap_num]
            check_control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in control_points_2d if p is not None])
            visualize_correspondences(check_control_points_3d, control_points_3d, 
                                      window_name=f"Backward, Innerframe, not allow errors, {len(check_control_points_3d)}/{len(control_points_3d)} points",
                                      background=(tgt_submap.get_points().reshape(-1, 3), tgt_submap.get_filtered_points_colors().reshape(-1, 3))
                                      )

        
        return control_points_2d, control_points_3d, control_points_idx, control_points_color
        


    def fit_tps_global(self):
        avg_control_points(self.control_points, self.submap_control_points, self.map, visualize=self.vis_tps, filter_method=self.filter_method, filter_k=self.filter_k, robust_mean_method=self.robust_mean_method)
        


    def optimize_inner_poses(self):
        obs = {
            cam_name: [] for cam_name in self.cam_names
        }
        ref2world_keep = {}
        for submap in self.map.get_submaps():
            poses = submap.org_poses
            assert poses.shape[0] % self.cam_num == 0
            ref2world_keep[submap.get_id()] = []
            for j in range(poses.shape[0]//self.cam_num):
                ref2world = poses[j*self.cam_num+self.ref_cam_id]
                ref2world_keep[submap.get_id()].append(ref2world)
                world2ref = se3_inv(ref2world)
                for i, cam_name in enumerate(self.cam_names):
                    obs[cam_name].append(se3_mul(world2ref, poses[i+j*self.cam_num]))

        for i in range(len(obs[self.ref_cam_name])):
            assert np.allclose(obs[self.ref_cam_name][i], np.eye(4), atol=1e-4)

        # print(analyze_rotation_bias(obs_dict=obs,
        #     cam_names=self.cam_names,
        #     axis_name=self.ref_cam_name,))
        # print(analyze_body_xyz_per_frame(obs_dict=obs,
        #     cam_names=self.cam_names,
        #     axis_name=self.ref_cam_name,))
        optimizer = RigOptimizerClosedForm(
            obs_dict=obs,
            cam_names=self.cam_names,
            axis_name=self.ref_cam_name,
        )
        optimizer._solve_one_shot()
        rig_poses = optimizer.get_corrected_per_frame_transforms()
        for submap_key in self.map.submaps.keys():
            submap = self.map.submaps[submap_key]
            assert submap.poses.shape[0] % self.cam_num == 0
            for j in range(submap.poses.shape[0]//self.cam_num):
                rig_poses_j = rig_poses[submap.get_id()]
                ref2world = ref2world_keep[submap.get_id()][j]
                for i in range(self.cam_num):
                    rig_poses_j[i] = se3_mul(ref2world, rig_poses_j[i])
                submap.poses[j*self.cam_num: (j+1)*self.cam_num] = rig_poses_j
                if not np.allclose(submap.poses[j*self.cam_num+self.ref_cam_id][:3,3], submap.org_poses[j*self.cam_num+self.ref_cam_id][:3,3], atol=1e-4):
                    ipdb.set_trace()

        # obs = {
        #     cam_name: [] for cam_name in self.cam_names
        # }
        # ref2world_keep = {}
        # for submap in self.map.get_submaps():
        #     if submap.get_id() == 0:
        #         poses = submap.poses
        #     else:
        #         poses = submap.poses[self.overlap_num:]
        #     assert poses.shape[0] % self.cam_num == 0
        #     ref2world_keep[submap.get_id()] = []
        #     for j in range(poses.shape[0]//self.cam_num):
        #         ref2world = poses[j*self.cam_num+self.ref_cam_id]
        #         ref2world_keep[submap.get_id()].append(ref2world)
        #         world2ref = se3_inv(ref2world)
        #         for i, cam_name in enumerate(self.cam_names):
        #             obs[cam_name].append(se3_mul(world2ref, poses[i+j*self.cam_num]))

        # for i in range(len(obs[self.ref_cam_name])):
        #     assert np.allclose(obs[self.ref_cam_name][i], np.eye(4), atol=1e-7)

        # print(analyze_rotation_bias(obs_dict=obs,
        #     cam_names=self.cam_names,
        #     axis_name=self.ref_cam_name,))
        # print(analyze_body_xyz_per_frame(obs_dict=obs,
        #     cam_names=self.cam_names,
        #     axis_name=self.ref_cam_name,))

    
    def write_poses_to_file(self, save_root):
        for submap in self.map.ordered_submaps_by_key():
            poses = submap.get_all_poses_world()
            frame_ids = submap.get_frame_ids()
            if len(poses) != len(frame_ids):
                frame_ids = frame_ids[self.overlap_num:self.overlap_num+self.submap_num]
            for frame_id, pose in zip(frame_ids, poses):
                save_path = os.path.join(save_root, frame_id+".txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(save_path, pose)

    # def save_framewise_pointclouds(self, save_root):
    #     os.makedirs(save_root, exist_ok=True)
    #     for submap in self.map.ordered_submaps_by_key():
    #         pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame()
    #         for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
    #             np.savez(f"{save_root}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)


    def save_framewise_pointclouds(self, save_root):
        os.makedirs(save_root, exist_ok=True)
        for submap in self.map.ordered_submaps_by_key():
            pcd = submap.get_filtered_points_in_world_frame()
            color = submap.get_filtered_points_colors()

            np.savez(f"{save_root}/{submap.get_id():0>2d}.npz", pcd=pcd, color=color)
