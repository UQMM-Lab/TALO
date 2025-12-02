import re
import os
import cv2
import ipdb
import torch
import numpy as np
import open3d as o3d

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.H_world_map = None
        self.R_world_map = None
        self.poses = None
        self.frames = None
        self.intrinscs = None
        self.retrieval_vectors = None
        self.colors = None # (S, H, W, 3)
        self.conf = None # (S, H, W)
        self.conf_threshold = None
        self.points = None # (S, H, W, 3)
        self.voxelized_points = None
        self.frame_ids = None
        self.org_poses = None
        self.save_world_points = None

    def add_all_poses(self, org_poses, poses):
        self.org_poses = org_poses
        self.poses = poses
    
    def get_poses(self):
        return self.poses
    
    def get_all_poses(self):
        return self.org_poses

    def add_all_conf(self, conf, conf_threshold, conf_all=None):
        self.conf = conf
        self.conf_threshold = conf_threshold
        self.mask = conf >= conf_threshold
        if conf_all is not None:
            self.mask_all = conf_all >= conf_threshold
        else:
            self.mask_all = self.mask
        # print(self.mask.sum() / self.mask.size, "points selected with conf threshold", conf_threshold)
    
    def filter_data_by_mask(self, data):
        return data[self.mask]
    
    def filter_all_data_by_mask(self, data):
        return data[self.mask_all]
    
    def get_mask(self):
        return self.mask

    def add_all_points(self, points, colors, intrinsics, colors_all=None, points_all=None, intrinsics_all=None):
        self.points = points
        self.colors = colors
        self.intrinscs = intrinsics
        if points_all is not None:
            self.points_all = points_all
        else:
            self.points_all = points
        if colors_all is not None:
            self.colors_all = colors_all
        else:
            self.colors_all = colors
        if intrinsics_all is not None:
            self.intrinscs_all = intrinsics_all
        else:
            self.intrinscs_all = intrinsics

    def get_intrinsics(self):
        return self.intrinscs
    
    def add_control_points(self, control_points_3d, control_points_2d):
        self.control_points_3d = control_points_3d
        self.control_points_2d = control_points_2d

    def get_control_points_3d(self):
        return self.control_points_3d

    def get_control_points_2d(self):
        return self.control_points_2d

    def add_all_frames(self, frames):
        self.frames = frames

    def get_HW(self):
        return self.frames.shape[-2:]
    
    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold
    
    def get_frame_at_index(self, index):
        return self.frames[index, ...]
    
    def get_frames(self):
        return self.frames

    def get_ref_frame(self, cam_num, ref_cam_id):
        return [self.frames[i] for i in range(len(self.frames)) if i % cam_num == ref_cam_id]

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    # def get_all_poses_world(self):
    #     projection_mat_list = self.vggt_intrinscs @ np.linalg.inv(self.poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
    #     poses = []
    #     for index, projection_mat in enumerate(projection_mat_list):
    #         cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
    #         # print("cal", cal/cal[2,2])
    #         trans = trans/trans[3,0] # TODO see if we should normalize the rotation too with this.
    #         pose = np.eye(4)
    #         pose[0:3, 0:3] = np.linalg.inv(rot)
    #         pose[0:3,3] = trans[0:3,0]
    #         poses.append(pose)
    #     return np.stack(poses, axis=0)

    def get_all_poses_world(self):
        projection_mat_list = self.intrinscs_all @ np.linalg.inv(self.org_poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
        poses = []
        for index, projection_mat in enumerate(projection_mat_list):
            cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
            # print("cal", cal/cal[2,2])
            trans = trans/trans[3,0] # TODO see if we should normalize the rotation too with this.
            pose = np.eye(4)
            pose[0:3, 0:3] = np.linalg.inv(rot)
            pose[0:3,3] = trans[0:3,0]
            poses.append(pose)
        return np.stack(poses, axis=0)
    
    def get_poses_world(self):
        projection_mat_list = self.intrinscs @ np.linalg.inv(self.poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
        poses = []
        for index, projection_mat in enumerate(projection_mat_list):
            cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
            # print("cal", cal/cal[2,2])
            trans = trans/trans[3,0] # TODO see if we should normalize the rotation too with this.
            pose = np.eye(4)
            pose[0:3, 0:3] = np.linalg.inv(rot)
            pose[0:3,3] = trans[0:3,0]
            poses.append(pose)
        return np.stack(poses, axis=0)
    
    def get_frame_pointcloud(self, pose_index):
        return self.points[pose_index]

    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names, 
        removing any leading zeros, and add them all to a list.

        Note: This does not include any of the loop closure frames.
        """
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map
    
    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def get_reference_homography(self):
        return self.H_world_map

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])
    
    def get_frame_ids(self):
        # Note this does not include any of the loop closure frames
        return self.frame_ids

    # def filter_data_by_mask(self, data):
    #     init_conf_mask = self.conf >= self.conf_threshold
    #     return data[init_conf_mask]
    

    # def get_points_list_in_world_frame(self):
    #     point_list = []
    #     frame_id_list = []
    #     frame_conf_mask = []
    #     for index,points in enumerate(self.pointclouds):
    #         points_flat = points.reshape(-1, 3)
    #         points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
    #         points_transformed = (self.H_world_map @ points_homogeneous.T).T
    #         point_list.append((points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape))
    #         frame_id_list.append(self.frame_ids[index])
    #         conf_mask = self.conf[index] >= self.conf_threshold
    #         frame_conf_mask.append(conf_mask)
    #     return point_list, frame_id_list, frame_conf_mask

    def get_points(self):
        return self.points

    def get_all_points(self):
        return self.points_all

    def get_filtered_points(self):
        return self.filter_data_by_mask(self.points)

    def get_all_filtered_points(self):
        return self.filter_all_data_by_mask(self.points_all)
    
    def get_points_in_world_frame(self):
        if self.save_world_points is not None:
            return self.save_world_points
        points = self.points
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        points_transformed = points_transformed[:, :3] / points_transformed[:, 3:]
        return points_transformed.reshape(self.points.shape)
    
    def get_filtered_points_in_world_frame(self):
        if self.save_world_points is not None:
            return self.save_world_points
        points = self.filter_data_by_mask(self.points)
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        return points_transformed[:, :3] / points_transformed[:, 3:]

    def get_all_filtered_points_in_world_frame(self):
        points = self.filter_all_data_by_mask(self.points_all)
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        return points_transformed[:, :3] / points_transformed[:, 3:]
        
    
    def set_points_in_world_frame(self, points_in_world):
        points_flat = points_in_world.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (np.linalg.inv(self.H_world_map) @ points_homogeneous.T).T
        self.points = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(self.points.shape)

    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                points = self.filter_data_by_mask(self.points)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_mask(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame
    
    def get_filtered_points_colors(self):
        colors = self.filter_data_by_mask(self.colors)
        return colors.reshape(-1, 3)

    def get_points_colors(self):
        colors = self.colors
        return colors

    def get_all_filtered_points_colors(self):
        colors = self.filter_all_data_by_mask(self.colors_all)
        return colors.reshape(-1, 3)