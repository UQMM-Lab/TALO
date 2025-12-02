import os
import pickle
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



from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

from rig_solver import *
from interframe_solver import *
from pi3.utils.geometry import homogenize_points, depth_edge
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from third_party_codes.vggt_code import *
from third_party_codes.pi3_code import load_images_as_tensor, recover_focal_no_shift
from third_party_codes.vggt_long_code import robust_weighted_estimate_sim3



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
        visualize_global_map: bool = False,
        interframe_solver_choice: str = "pose",
        gradio_mode: bool = False,
        port = 8080,
        save_root = "",
        overlap_size = 1,
        submap_size = 1,
        cam_names = [],
        ref_cam_id = -1,
        ref_cam_name = '',
        vis_tps=False,
        allow_nonvalid=False,
        voxel_size_reate=0.05,
        max_project_error=0.1,
        filter_method="mean",
        filter_k=32,
        robust_mean_method=None,
        backward_control_points=False,
        ):
        self.model = model
        if model == "VGGT":
            from vggt.utils.load_fn import load_and_preprocess_images
            self.load_and_preprocess_images = load_and_preprocess_images
        elif model == "Pi3":
            self.load_and_preprocess_images = load_images_as_tensor
        elif model == "MapAnything":
            from mapanything.utils.image import load_images
            self.load_and_preprocess_images = load_images
        self.init_conf_threshold = init_conf_threshold
        self.gradio_mode = gradio_mode
        self.save_root = save_root

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
        self.voxel_size_rate = voxel_size_reate
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
        extrinsics = submap.get_poses_world()  # (S, 3, 4)
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_frames()
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
        """
        # Unpack prediction dict
        colors = pred_dict["images"]  # (S, H, W, 3)
        detected_loops = pred_dict["detected_loops"]
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        cam_to_world = pred_dict["cam2world"]  # (S, 4, 4)
        world_points = pred_dict["world_points"]  # (S, H, W, 3)
        conf = pred_dict["world_points_conf"]  # (S, H, W)

        new_pcd_num = self.current_working_submap.get_id()
        self.submap_control_points[self.current_working_submap.get_id()] = []
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[self.submap_num:self.submap_num+self.overlap_num]
            self.prior_conf = conf[self.submap_num:self.submap_num+self.overlap_num]
            self.prior_poses = cam_to_world[self.submap_num:self.submap_num+self.overlap_num]

            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
            
            self.current_working_submap.set_reference_homography(H_w_submap)
            self.current_working_submap.add_all_poses(cam_to_world, cam_to_world.copy())
            self.current_working_submap.add_all_points(world_points, colors, intrinsics_cam)
            self.current_working_submap.add_all_conf(conf, np.percentile(conf[conf>0], self.init_conf_threshold))
            self.map.add_submap(self.current_working_submap)

        else:
            # visualize_correspondences(world_points[:self.overlap_num].reshape(-1, 3),
            #                           world_points[self.overlap_num:self.overlap_num+self.submap_num].reshape(-1, 3))
            prior_sid = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_sid)
            self.current_working_submap.add_all_conf(conf[self.overlap_num:self.overlap_num+self.submap_num], np.percentile(conf[conf>0], self.init_conf_threshold), conf)
            if pred_dict.get('valid_mask') is not None:
                self.current_working_submap.add_all_mask(pred_dict['valid_mask'][self.overlap_num:self.overlap_num+self.submap_num])

            current_pts = world_points[:self.overlap_num]
        
            # TODO conf should be using the threshold in its own submap
            good_mask = (self.prior_conf >= prior_submap.get_conf_threshold()) & (conf[:self.overlap_num] >= self.current_working_submap.get_conf_threshold())
            # good_mask1 = self.prior_conf > prior_submap.get_conf_threshold() * (conf[:self.overlap_num] > prior_submap.get_conf_threshold())
            # print(good_mask1.sum()/good_mask1.size, good_mask.sum()/good_mask.size)

            if self.interframe_solver_choice.startswith("pose"):
                current_poses = cam_to_world[:self.overlap_num]
                H_relative_ls = []
                for i, (prior_pose, current_pose) in enumerate(zip(self.prior_poses, current_poses)):
                    if i % self.cam_num == self.ref_cam_id:
                        H_relative = prior_pose @ np.linalg.inv(current_pose)
                        H_relative_ls.append(H_relative)
                    if len(H_relative_ls) == 4:
                        break
                # if len(H_relative_ls) > 1:
                #     H_relative_ls = np.array(H_relative_ls) # (N, 4, 4)
                #     H_relative_R = H_relative_ls[:, :3, :3] # (N, 3, 3)
                #     H_relative_R = chordal_L2_mean_rotation(H_relative_R)
                #     H_relative_t = H_relative_ls[:, :3, 3]  # (N, 3)
                #     H_relative_t = H_relative_t.mean(axis=0)
                #     H_relative = np.eye(4)
                #     H_relative[:3, :3] = H_relative_R
                #     H_relative[:3,  3] = H_relative_t
                # else:
                H_relative = H_relative_ls[0]

                if self.model != "MapAnything":

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

                    s = np.clip(s, 0.7, 1.3)
                    # print(colored("scale factor", 'green'), s)
                    world_points *= s
                    current_pts = world_points[:self.overlap_num]
                    cam_to_world[:, 0:3, 3] *= s

                    current_poses = cam_to_world[:self.overlap_num]
                    H_relative_ls = []
                    for i, (prior_pose, current_pose) in enumerate(zip(self.prior_poses, current_poses)):
                        if i % self.cam_num == self.ref_cam_id:
                            H_relative = prior_pose @ np.linalg.inv(current_pose)
                            H_relative_ls.append(H_relative)
                    if len(H_relative_ls) > 1:
                        H_relative_ls = np.array(H_relative_ls) # (N, 4, 4)
                        H_relative_R = H_relative_ls[:, :3, :3] # (N, 3, 3)
                        H_relative_R = chordal_L2_mean_rotation(H_relative_R)
                        H_relative_t = H_relative_ls[:, :3, 3]  # (N, 3)
                        H_relative_t = H_relative_t.mean(axis=0)
                        H_relative = np.eye(4)
                        H_relative[:3, :3] = H_relative_R
                        H_relative[:3,  3] = H_relative_t
                    else:
                        H_relative = H_relative_ls[0]

            elif self.interframe_solver_choice.startswith("sim3"):
                s, R, t = robust_weighted_estimate_sim3(
                    src=current_pts[good_mask], 
                    tgt=self.prior_pcd[good_mask],
                    init_weights=np.sqrt(self.prior_conf[good_mask] * conf[:self.overlap_num][good_mask]),
                    use_scale=self.model!="MapAnything"
                )
                H_relative = np.eye(4)
                H_relative[:3, :3] = R
                H_relative[:3,  3] = t
                world_points *= s
                current_pts = world_points[:self.overlap_num]

                cam_to_world[:, 0:3, 3] *= s
                
            elif self.interframe_solver_choice == "sl4":
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            #########################################################
            #########################################################
            #########################################################

            # TODO
            # src = world_points[:self.overlap_num][good_mask]
            # tgt = self.prior_pcd[good_mask]
            # color = colors[:self.overlap_num][good_mask]
            # pred = apply_homography(H_relative, src)
            # src_other = world_points[self.overlap_num:self.overlap_num+self.submap_num][conf[self.overlap_num:self.overlap_num+self.submap_num] > self.current_working_submap.get_conf_threshold()]
            # src_other_color = colors[self.overlap_num:self.overlap_num+self.submap_num][conf[self.overlap_num:self.overlap_num+self.submap_num] > self.current_working_submap.get_conf_threshold()]
            # pred_other = apply_homography(H_relative, src_other)
            # tgt_other = prior_submap.pointclouds[:self.overlap_num][prior_submap.mask[:self.overlap_num]]
            # tgt_other_color = prior_submap.colors[:self.overlap_num][prior_submap.mask[:self.overlap_num]]

            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            # tgt = apply_homography(prior_submap.get_reference_homography(), tgt)
            # pred = apply_homography(H_w_submap, src)
            # save_vis_path = os.path.join(self.save_root, 'align_vis', f'submap_{new_pcd_num:0>2d}_to_{prior_sid:0>2d}')
            # os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
            # np.savez(save_vis_path, 
            #          tgt=tgt, pred=pred)


            self.prior_pcd = world_points[:self.overlap_num+self.submap_num][-self.overlap_num:]
            self.prior_conf = conf[:self.overlap_num+self.submap_num][-self.overlap_num:]
            self.prior_poses = cam_to_world[:self.overlap_num+self.submap_num][-self.overlap_num:]

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)
            # Add between factor.
            self.graph.add_between_factor(prior_sid, new_pcd_num, H_relative, self.graph.relative_noise)


            # print("added between factor", prior_pcd_num, new_pcd_num)

            self.current_working_submap.set_reference_homography(H_w_submap)
            self.current_working_submap.add_all_poses(cam_to_world[:self.overlap_num+self.submap_num], cam_to_world[self.overlap_num:self.overlap_num+self.submap_num])
            self.current_working_submap.add_all_points(world_points[self.overlap_num:self.overlap_num+self.submap_num], 
                                                       colors[self.overlap_num:self.overlap_num+self.submap_num], 
                                                       intrinsics_cam[self.overlap_num:self.overlap_num+self.submap_num],
                                                       points_all=world_points[:self.overlap_num+self.submap_num],
                                                       colors_all=colors[:self.overlap_num+self.submap_num],
                                                       intrinsics_all=intrinsics_cam[:self.overlap_num+self.submap_num],
                                                       )
            
        
            self.map.add_submap(self.current_working_submap)
            # self.optimize_inner_poses()
            if "tps" in self.interframe_solver_choice:
                pts = current_pts[good_mask]
                pts = pts[np.isfinite(pts).all(axis=1)]
                center = np.mean(pts, axis=0)  # 更鲁棒的中心
                dists = np.linalg.norm(pts - center, axis=1)
                scene_radius = np.quantile(dists, 0.98)

                voxel_size = 0.05#scene_radius * self.voxel_size_rate
                max_error = 0.05#scene_radius * self.max_project_error

                # voxel_size = 1#scene_radius * self.voxel_size_rate
                # max_error = 1
                print(f"Estimated scene radius: {scene_radius:.3f}, voxel size: {voxel_size:.3f} (≈{(self.voxel_size_rate)*100:.1f}%), max projection error: {max_error:.3f} (≈{(self.max_project_error)*100:.1f}% )")

                H, W = self.current_working_submap.get_HW()

                old_control_points_idx, old_control_points_3d = self.forward_control_points(prior_submap.get_id(), current_pts)

                # —— 控制点选择 ——
                old_control_points_keep_idx, new_control_points_idx = select_control_points_np(
                    current_pts[good_mask],  # 选择阶段用 float32 即可
                    method="voxel",         # "random" | "voxel" | "octree"
                    kwargs={"voxel_size":voxel_size, "min_threshold":1},  # 比如 {"voxel_size":0.1} 或 {"max_depth":7,"leaf_points":256}
                    # vis=True,
                    given_control_points=old_control_points_3d,
                    keep_all_given=self.allow_nonvalid
                )
                print(colored(f"Found {len(old_control_points_keep_idx)} old control points, {len(new_control_points_idx)} new control points", "cyan"))
                cam_to_world_ = self.current_working_submap.poses[-self.overlap_num:]
                intrinsics_cam_ = intrinsics_cam[:self.overlap_num+self.submap_num][-self.overlap_num:]
                current_pts_ = world_points[:self.overlap_num+self.submap_num][-self.overlap_num:]
                valid_mask = conf[:self.overlap_num+self.submap_num][-self.overlap_num:]>=self.current_working_submap.get_conf_threshold()
                if old_control_points_keep_idx.shape[0] > 0:
                    old_control_points_3d = old_control_points_3d[old_control_points_keep_idx]
                    old_control_points_2d = project(old_control_points_3d, cam_to_world_, intrinsics_cam_, current_pts_, H, W, valid_mask=valid_mask if not self.allow_nonvalid else None, max_error=max_error)
                    if self.vis_tps:
                        check_3d = np.array([current_pts_[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
                        check_3d_2 = np.array([old_control_points_3d[i] for i in range(len(old_control_points_2d)) if old_control_points_2d[i] is not None])
                        visualize_correspondences(check_3d, check_3d_2, window_name="Old Innerframe, not allow errors", background=(current_pts[good_mask], colors[:self.overlap_num][good_mask]))
                    for i, point_id in enumerate(old_control_points_keep_idx):
                        self.control_points[old_control_points_idx[point_id]][self.current_working_submap.get_id()] = {
                            "3d": old_control_points_3d[i],
                            '2d': old_control_points_2d[i]
                        }
                    self.submap_control_points[self.current_working_submap.get_id()] += [old_control_points_idx[i] for i in old_control_points_keep_idx]

                new_control_points_3d = current_pts[good_mask][new_control_points_idx]
                new_control_points_2d = project(new_control_points_3d, cam_to_world_, intrinsics_cam_, current_pts_, H, W, valid_mask=valid_mask if not self.allow_nonvalid else None, max_error=max_error)
                if self.vis_tps:
                    check_3d = np.array([current_pts_[p[0], p[2], p[1]] for p in new_control_points_2d if p is not None])
                    if len(check_3d) > 0:
                        check_3d_2 = np.array([new_control_points_3d[i] for i in range(len(new_control_points_2d)) if new_control_points_2d[i] is not None])
                        visualize_correspondences(check_3d, check_3d_2, window_name="New Innerframe, not allow errors", background=(world_points[conf>=self.current_working_submap.get_conf_threshold()], colors[conf>=self.current_working_submap.get_conf_threshold()]))
                start_idx = len(self.control_points)
                for point_id in range(new_control_points_3d.shape[0]):
                    self.control_points[start_idx + point_id] = {
                            self.current_working_submap.get_id(): {
                                "3d": new_control_points_3d[point_id],
                                "2d": new_control_points_2d[point_id]
                            }
                    }
                self.submap_control_points[self.current_working_submap.get_id()] += [start_idx + i for i in range(new_control_points_3d.shape[0])]


                # backward new control points to all previous submaps
                orig_idx = np.flatnonzero(good_mask)[new_control_points_idx]
                xs, ys, zs = np.unravel_index(orig_idx, self.prior_conf.shape)
                new_control_points_2d = np.stack([xs, zs, ys], axis=1)
                
                new_control_points_idx = [start_idx + i for i in range(new_control_points_3d.shape[0])]

                submap_id = -1
                for submap_id in list(self.map.submaps.keys())[::-1][1:]:
                    new_control_points_2d, new_control_points_3d, new_control_points_idx = self.backward_control_points(
                        old_control_points_2d=new_control_points_2d,
                        old_control_points_3d=new_control_points_3d,
                        old_control_points_idx=new_control_points_idx,
                        tgt_submap_id=submap_id,
                        max_error=max_error
                    )
                    if len(new_control_points_idx) == 0 or not self.backward_cps:
                        break
                # print(colored(f"stop backward at submap {submap_id} for {self.current_working_submap.get_id()} due to no control points", "yellow"))
            
        
        

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
        images = self.load_and_preprocess_images(image_names)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model == "MapAnything":
            images_add = torch.cat([image['img'] for image in images])
        else:
            images = images.to(device)
            images_add = images
        
        # Check for loop closures
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images_add)
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
            # not implemented for mapanything yet
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            # new_submap.add_all_frames(images)

        self.current_working_submap = new_submap
        results = {}
        if self.model == "MapAnything":
            non_sky_mask_binary = apply_sky_segmentation(image_names, images[0]['true_shape'][0].tolist())  # (n, H, W)
            predictions = model.infer(
                images,                            # Input views
                memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
                use_amp=True,                     # Use mixed precision inference (recommended)
                amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
                apply_mask=True,                  # Apply masking to dense geometry outputs
                mask_edges=True,                  # Remove edge artifacts by using normals and depth
                apply_confidence_mask=False,      # Filter low-confidence regions
                confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
            )
            results = {
                "cam2world": torch.cat([pred["camera_poses"] for pred in predictions], 0).cpu().numpy(),         # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
                "intrinsic": torch.cat([pred["intrinsics"] for pred in predictions], 0).cpu().numpy(),           # Recovered pinhole camera intrinsics (B, 3, 3)
                "world_points": torch.cat([pred["pts3d"] for pred in predictions], 0).cpu().numpy(),                     # 3D points in world coordinates (B, H, W, 3)
                "world_points_conf": torch.cat([(pred["conf"] * pred["mask"].squeeze(-1)) for pred in predictions], 0).cpu().numpy(),                   # Per-pixel confidence scores (B, H, W)
                "images": (torch.cat([pred["img_no_norm"] for pred in predictions], 0).cpu().numpy()*255).astype(np.uint8)                  # Denormalized input images for visualization (B, H, W, 3)
            }
            results['cam2world'] = np.linalg.inv(results['cam2world'][0]) @ results['cam2world'] # (S, 4, 4)
            results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]
            results["detected_loops"] = detected_loops
        else:
            non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])  # (n, H, W)
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            if self.model == "Pi3":
                images = images[None]
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            if self.model == "VGGT":
                results["images"] = (images.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                results["cam2world"] = closed_form_inverse_se3(extrinsic.cpu().numpy().squeeze(0))
                results["intrinsic"] = intrinsic.cpu().numpy().squeeze(0)
                results["detected_loops"] = detected_loops
                results['world_points'] = predictions["world_points"].cpu().numpy().squeeze(0)  # (S, H, W, 3)
                results["world_points_conf"] = (non_sky_mask_binary * predictions["world_points_conf"].cpu().numpy().squeeze(0))
            elif self.model == "Pi3":
                cam2world = torch.inverse(predictions['camera_poses'][0][0]) @ predictions['camera_poses'][0] # (S, 4, 4)
                results['cam2world'] = cam2world.cpu().numpy() # (S, 4, 4)
                results['world_points'] = torch.einsum('nij, nhwj -> nhwi', cam2world, homogenize_points(predictions['local_points'][0]))[..., :3].cpu().numpy() # (S, H, W, 3)
                results['world_points_conf'] = torch.sigmoid(predictions['conf'][0,...,0]).cpu().numpy() # (S, H, W)
                results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]
                results["images"] = (images[0].permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
                results["detected_loops"] = detected_loops


                points = predictions["local_points"] # (1, S, H, W, 3)
                masks = None
                original_height, original_width = points.shape[-3:-1]
                aspect_ratio = original_width / original_height
                # use recover_focal_shift function from MoGe
                focal = recover_focal_no_shift(points, masks) # focal: (1, S), shift: (1, S)
                fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
                zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
                zero_point_five = torch.full_like(fx, 0.5)
                intrinsics = torch.stack([
                    fx, zeros, zero_point_five, 
                    zeros, fy, zero_point_five, 
                    zeros, zeros, ones
                ], dim=-1).reshape(-1, 3, 3) # (S, 3, 3)
                
                
                H, W = original_height, original_width

                # 归一化 -> 像素（乘以各向尺寸-1）
                intrinsics[:, 0, 0] *= (W - 1)   # fx
                intrinsics[:, 1, 1] *= (H - 1)   # fy
                intrinsics[:, 0, 2] *= (W - 1)   # cx
                intrinsics[:, 1, 2] *= (H - 1)   # cy
                results['intrinsic'] = intrinsics.cpu().numpy()
                # results['valid_mask'] = ~depth_edge(predictions['local_points'][..., 2], rtol=0.03)[0].cpu().numpy() # (S, H, W)

        return results



    def forward_control_points(self, src_sid, points_3d):
        old_control_points_2d_dict = {}
        old_control_points_3d_check = []
        for point_id in self.submap_control_points.get(src_sid, []):
            point2d = self.control_points[point_id][src_sid]["2d"]
            if point2d is not None:
                old_control_points_2d_dict[point_id] = point2d
                old_control_points_3d_check.append(self.control_points[point_id][src_sid]["3d"])
        if len(old_control_points_3d_check) > 0:
            old_control_points_3d_check = np.stack(old_control_points_3d_check, axis=0)
            old_control_points_2d = list(old_control_points_2d_dict.values())
            old_control_points_idx = list(old_control_points_2d_dict.keys())
            old_control_points_3d = np.array([points_3d[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
            if self.vis_tps:
                visualize_correspondences(old_control_points_3d, old_control_points_3d_check, window_name="Interframe, allow errors")
        else:
            old_control_points_3d = None
            old_control_points_idx = None
        return old_control_points_idx, old_control_points_3d

    def backward_control_points(self, old_control_points_2d, old_control_points_3d, old_control_points_idx, tgt_submap_id, max_error):
        # tgt_submap = self.map.get_submap(tgt_submap_id)
        # cam_to_world = tgt_submap.get_poses_world()  # (S, 3, 4)
        # intrinsics_cam = tgt_submap.get_all_intrinsics()  # (S, 3, 3)
        # valid_mask = tgt_submap.get_all_conf() >= tgt_submap.get_conf_threshold()  # (S, H, W)
        # pts_3d = tgt_submap.get_all_points()[:self.overlap_num+self.submap_num][-self.overlap_num:]  

        # if not self.allow_nonvalid and tgt_submap_id < self.current_working_submap.get_id()-1:
        #     old_control_points_2d_tp = []
        #     old_control_points_idx_tp = []
        #     valid_mask_tp = valid_mask[:self.overlap_num+self.submap_num][-self.overlap_num:]
        #     for i, p in enumerate(old_control_points_2d):
        #         if valid_mask_tp[p[0], p[2], p[1]]:
        #             old_control_points_2d_tp.append(p)
        #             old_control_points_idx_tp.append(old_control_points_idx[i])
        #     # if tgt_submap_id == self.current_working_submap.get_id()-1:
        #     #     assert len(old_control_points_2d) == len(old_control_points_2d_tp)
        #     old_control_points_2d = old_control_points_2d_tp
        #     old_control_points_idx = old_control_points_idx_tp

        # control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
        # if self.vis_tps:
        #     visualize_correspondences(control_points_3d, old_control_points_3d, window_name=f"Backward, Interframe, allow errors, {len(control_points_3d)}/{len(old_control_points_3d)} points")

        # H, W = tgt_submap.get_HW()
        # control_points_2d = project(control_points_3d, cam_to_world[:self.overlap_num], intrinsics_cam[:self.overlap_num], pts_3d[:self.overlap_num], H, W, valid_mask=valid_mask[:self.overlap_num] if not self.allow_nonvalid else None, max_error=self.max_project_error)
        # for i, point_id in enumerate(old_control_points_idx):
        #     self.control_points[point_id][tgt_submap_id] = {
        #         "3d": control_points_3d[i],
        #         # '2d': control_points_2d[i]
        #     }
        # self.submap_control_points[tgt_submap_id].extend(old_control_points_idx)
        
        # if tgt_submap_id == 0:
        #     return [], [], []
        
        # control_points_3d = np.array([control_points_3d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None])
        # control_points_idx = [old_control_points_idx[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None]
        # control_points_2d = [control_points_2d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None]
        # if self.vis_tps:
        #     pts_3d = tgt_submap.get_all_points()[:self.overlap_num]
        #     check_control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in control_points_2d if p is not None])
        #     visualize_correspondences(check_control_points_3d, control_points_3d, 
        #                               window_name=f"Backward, Innerframe, not allow errors, {len(check_control_points_3d)}/{len(control_points_3d)} points",
        #                               background=(tgt_submap.get_points().reshape(-1, 3), tgt_submap.get_filtered_points_colors().reshape(-1, 3))
        #                               )

        # return control_points_2d, control_points_3d, control_points_idx


        tgt_submap = self.map.get_submap(tgt_submap_id)
        cam_to_world = tgt_submap.get_poses_world()  # (S, 3, 4)
        intrinsics_cam = tgt_submap.get_intrinsics()  # (S, 3, 3)
        valid_mask = tgt_submap.get_mask() # (S, H, W)
        pts_3d = tgt_submap.get_all_points()[:self.overlap_num+self.submap_num][-self.overlap_num:]  

        if not self.allow_nonvalid and tgt_submap_id < self.current_working_submap.get_id()-1:
            old_control_points_2d_tp = []
            old_control_points_idx_tp = []
            old_control_points_3d_tp = []
            valid_mask_tp = valid_mask[:self.overlap_num+self.submap_num][-self.overlap_num:]
            for i, p in enumerate(old_control_points_2d):
                if valid_mask_tp[p[0], p[2], p[1]]:
                    old_control_points_2d_tp.append(p)
                    old_control_points_idx_tp.append(old_control_points_idx[i])
                    old_control_points_3d_tp.append(old_control_points_3d[i])
            # if tgt_submap_id == self.current_working_submap.get_id()-1:
            #     assert len(old_control_points_2d) == len(old_control_points_2d_tp)
            old_control_points_2d = np.array(old_control_points_2d_tp)
            old_control_points_idx = old_control_points_idx_tp
            old_control_points_3d = np.array(old_control_points_3d_tp)

        control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in old_control_points_2d if p is not None])
        if self.vis_tps:
            visualize_correspondences(control_points_3d, old_control_points_3d, window_name=f"Backward, Interframe, allow errors, {len(control_points_3d)}/{len(old_control_points_3d)} points")

        for i, point_id in enumerate(old_control_points_idx):
            self.control_points[point_id][tgt_submap_id] = {
                "3d": control_points_3d[i],
                # '2d': control_points_2d[i]
            }
        self.submap_control_points[tgt_submap_id].extend(old_control_points_idx)
        if tgt_submap_id == 0 or len(control_points_3d) == 0:
            return [], [], []

        H, W = tgt_submap.get_HW()
            
        pts_3d = tgt_submap.get_all_points()
        control_points_2d = project(control_points_3d, cam_to_world[:self.overlap_num], intrinsics_cam[:self.overlap_num], pts_3d[:self.overlap_num], H, W, valid_mask=valid_mask[:self.overlap_num] if not self.allow_nonvalid else None, max_error=max_error)
        
        control_points_3d = np.array([control_points_3d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None])
        if len(control_points_3d) == 0:
            return [], [], []
        control_points_idx = [old_control_points_idx[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None]
        control_points_2d = np.array([control_points_2d[i] for i in range(len(control_points_2d)) if control_points_2d[i] is not None])
        if self.vis_tps:
            check_control_points_3d = np.array([pts_3d[p[0], p[2], p[1]] for p in control_points_2d if p is not None])
            visualize_correspondences(check_control_points_3d, control_points_3d, 
                                    window_name=f"Backward, Innerframe, not allow errors, {len(check_control_points_3d)}/{len(control_points_3d)} points",
                                    background=(tgt_submap.get_filtered_points().reshape(-1, 3), tgt_submap.get_filtered_points_colors().reshape(-1, 3))
                                    )

        
        return control_points_2d, control_points_3d, control_points_idx
        


    def fit_tps_global(self):
        reference_homographys = {sid: self.map.get_submap(sid).get_reference_homography() for sid in self.map.submaps.keys()}
        points_in_world_frames = {sid: self.map.get_submap(sid).get_all_filtered_points_in_world_frame() for sid in self.map.submaps.keys()}
        colors = {sid: self.map.get_submap(sid).get_all_filtered_points_colors()/255.0 for sid in self.map.submaps.keys()}
        poses = {sid: self.map.get_submap(sid).get_all_poses_world() for sid in self.map.submaps.keys()}
        save_path = os.path.join(self.save_root, "control_points_info.pkl")
        data = {
            "control_points": self.control_points,
            "submap_control_points": self.submap_control_points,
            "reference_homographys": reference_homographys,
            "points_in_world_frames": points_in_world_frames,
            "colors": colors,
            "poses": poses
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        # new_points_in_world_frames = avg_control_points(self.control_points, self.submap_control_points, 
        #                    reference_homographys, points_in_world_frames, colors,
        #                    visualize=self.vis_tps, filter_method=self.filter_method, filter_k=self.filter_k, robust_mean_method=self.robust_mean_method)
        # for submap_key in self.map.submaps.keys():
        #     submap = self.map.submaps[submap_key]
        #     submap.save_world_points = new_points_in_world_frames[submap_key]




    def optimize_inner_poses(self):
        if self.cam_num == 1:
            return
        obs_cam2world = []
        for submap in self.map.get_submaps():
            poses = submap.org_poses
            assert poses.shape[0] % self.cam_num == 0
            for i in range(poses.shape[0]//self.cam_num):
                obs_cam2world.append([])
                ref2world = poses[i*self.cam_num+self.ref_cam_id]
                world2ref = np.linalg.inv(ref2world)
                for j in range(self.cam_num):
                    obs_cam2world[-1].append(world2ref @ poses[i*self.cam_num + j])
                    
        for cam2world in obs_cam2world:
            assert np.allclose(cam2world[self.ref_cam_id], np.eye(4), atol=1e-5)


        rig_poses = cal_rig(np.array(obs_cam2world))
        assert np.allclose(rig_poses[self.ref_cam_id], np.eye(4), atol=1e-5)

        for submap_key in self.map.submaps.keys():
            submap = self.map.submaps[submap_key]
            assert submap.poses.shape[0] % self.cam_num == 0
            for i in range(submap.poses.shape[0]//self.cam_num):
                ref2world = submap.poses[i*self.cam_num + self.ref_cam_id]
                for j in range(self.cam_num):
                    submap.poses[i*self.cam_num + j] = ref2world @ rig_poses[j]
            for i in range(submap.org_poses.shape[0]//self.cam_num):
                ref2world = submap.org_poses[i*self.cam_num + self.ref_cam_id]
                for j in range(self.cam_num):
                    submap.org_poses[i*self.cam_num + j] = ref2world @ rig_poses[j]



    def write_to_each_camera(self):
        for submap in self.map.ordered_submaps_by_key():
            poses = submap.get_poses_world()
            masks = submap.get_mask()
            frame_ids = submap.get_frame_ids()
            points_world = submap.get_points_in_world_frame()
            points_color = submap.get_points_colors()
            if len(frame_ids) != len(poses):
                frame_ids = frame_ids[self.overlap_num:self.overlap_num+self.submap_num]
            for frame_id, pose, mask, point, color in zip(frame_ids, poses, masks, points_world, points_color):
                save_path = os.path.join(self.save_root, frame_id+".npz")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, pose=pose, point=point, color=color, mask=mask)


    def save_framewise_pointclouds(self):
        return
        # os.makedirs(self.save_root, exist_ok=True)
        # for submap in self.map.ordered_submaps_by_key():
        #     world_poses_all = submap.get_all_poses_world()
        #     world_points_all_filtered = submap.get_all_filtered_points_in_world_frame() # for
        #     colors_all_filtered = submap.get_all_filtered_points_colors()
            
        #     local_poses_all = submap.get_all_poses()
        #     local_points_all_filtered = submap.get_all_filtered_points()

        #     np.savez(f"{self.save_root}/{submap.get_id():0>3d}.npz", 
        #              world_points=world_points_all_filtered, 
        #              world_poses=world_poses_all,  
        #              local_points=local_points_all_filtered, 
        #              local_poses=local_poses_all,
        #              color=colors_all_filtered, 
        #         )
