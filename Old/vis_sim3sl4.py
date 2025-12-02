import ipdb
import numpy as np
import open3d as o3d
import os
import sys
from eval_vis_pcd_traj import create_camera_frustum, umeyama_alignment

def vis_sl4(src, tgt, pred, color, src_other=None, tgt_other=None, pred_other=None, src_other_color=None, tgt_other_color=None):

    def to_pcd(pts, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if isinstance(color, list):
            pcd.paint_uniform_color(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(color/255)
        return pcd

    # 组A（左）：原始 src vs tgt（不平移）
    shift = np.array([[0, 0, 5]])
    src = src + shift  # 整体 +Z 平移，避免遮挡
    src_other = src_other + shift
    pcd_src = to_pcd(np.concatenate((src, src_other), axis=0), np.concatenate((color, src_other_color), axis=0))  # 浅蓝色表示源点云
    pcd_tgt = to_pcd(np.concatenate((tgt, tgt_other), axis=0), np.concatenate((color, tgt_other_color), axis=0))  # 浅灰色表示目标点云

    # 组B（右）：变换后 src vs tgt（整体 +X 平移）
    pcd_pred = to_pcd(np.concatenate((pred, pred_other), axis=0), np.concatenate((color, src_other_color), axis=0))

    def scale_to_range(arr, new_min=0.25, new_max=1.0):
        old_min = arr.min()
        old_max = arr.max()
        return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # ===== 带误差渐变色的连线 =====
    def make_lines_colored_by_error(src_pts, tgt_pts, cmap_name="coolwarm"):
        assert src_pts.shape == tgt_pts.shape
        N = src_pts.shape[0]

        # 计算误差（L2 距离）
        d = np.linalg.norm(src_pts - tgt_pts, axis=1)  # (N,)
        d_norm = -scale_to_range(d, -1, .0)

        # 使用 matplotlib colormap（turbo 或 viridis 都适合论文）
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
        colors = cmap(d_norm)[:, :3]  # RGBA → RGB

        pts = np.vstack([src_pts, tgt_pts])  # (2N,3)
        lines = np.array([[i, i + N] for i in range(N)], dtype=np.int32)

        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        return ls

    random_keep_line_index = np.random.choice(src.shape[0], size=src.shape[0]//20, replace=False) # 
    # 左组连线（src vs tgt before）
    lines_A = make_lines_colored_by_error(src[random_keep_line_index], tgt[random_keep_line_index])
    lines_B = make_lines_colored_by_error(pred[random_keep_line_index], tgt[random_keep_line_index])



    return [pcd_src, pcd_tgt], [pcd_pred, pcd_tgt, lines_B]



if __name__ == "__main__":
    root = '/home/fengyi/Proj/VGGT-SLAM/Save/nuscenes' # sys.argv[1]
    gt_root = '/home/fengyi/Proj/VGGT-SLAM/Data/nuscenes'
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    height, width = 900, 1600
    
    root = '/home/fengyi/Proj/VGGT-SLAM/Save/waymo_less' # sys.argv[1]
    gt_root = '/home/fengyi/Proj/VGGT-SLAM/Data/waymo_less'
    cams = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT']#, 'SIDE_LEFT', 'SIDE_RIGHT']
    height, width = 1280, 1920

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="All Control Points")
    opt = vis.get_render_option()
    opt.point_size = float(5.0)
    opt.light_on = False
    # 键盘：N 继续；Q 退出（可选）
    state = {"step": False, "quit": False}
    vis.register_key_callback(ord("N"), lambda _: state.__setitem__("step", True))
    vis.register_key_callback(ord("Q"), lambda _: state.__setitem__("quit", True))

    first = True  # 首帧重置包围盒，后续保持视角

    for scene_dir in sorted(os.listdir(root)):
        scene_path = os.path.join(root, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        for model_dir in sorted(os.listdir(scene_path)):
            if 'Pi3' not in model_dir:
                continue
            model_path = os.path.join(scene_path, model_dir)
            for file in sorted(os.listdir(os.path.join(model_path, 'align_vis'))):
                if not file.endswith('.npz'):
                    continue
                this, prev = int(file.split('.')[0].split('_')[1]), int(file.split('.')[0].split('_')[3])
                # new_name = f'submap_{this:0>2d}_to_{prev:0>2d}.npz'
                # if file != new_name:
                #     os.rename(os.path.join(model_path, 'sl4_vis', file), os.path.join(model_path, 'sl4_vis', new_name))
                # print(f"Renamed {file} to {new_name}")
                # file = new_name
                # continue
                data = np.load(os.path.join(model_path, 'align_vis', file))
                geometries = vis_sl4(data['src'], data['tgt'], data['pred'], data['color'], data['src_other'], data['tgt_other'], data['pred_other'], data['src_other_color'], data['tgt_other_color'])


                intrinsics = [np.loadtxt(os.path.join(gt_root, scene_dir, 'intrinsic', f"{cams[i]}.txt")) for i in range(len(cams))]
                gt_prev_frame = [np.loadtxt(os.path.join(model_path, 'aligned_cam2world_gt', f"{cam_id}/{prev:0>3d}.txt")) for cam_id, cam in enumerate(cams)]
                gt_prev_overlap_frame = [np.loadtxt(os.path.join(model_path, 'aligned_cam2world_gt', f"{cam_id}/{prev*2:0>3d}.txt")) for cam_id, cam in enumerate(cams)]
                gt_this_overlap_frame = [np.loadtxt(os.path.join(model_path, 'aligned_cam2world_gt', f"{cam_id}/{this:0>3d}.txt")) for cam_id, cam in enumerate(cams)]
                gt_this_frame = [np.loadtxt(os.path.join(model_path, 'aligned_cam2world_gt', f"{cam_id}/{this*2:0>3d}.txt")) for cam_id, cam in enumerate(cams)]
                
                pred_prev_frame = np.load(os.path.join(model_path, f'{prev:0>3d}.npz'))['poses']
                pred_this_frame = np.load(os.path.join(model_path, f'{this:0>3d}.npz'))['poses']
                # pred_this_frame = pred_this_frame + np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,5,1]]])  # +Z 平移，避免遮挡

                # pred_cam2world = [np.loadtxt(os.path.join(model_path, cams[0], f"{idx:0>3d}.txt")) for idx in range(40)]  # 40 帧
                # gt_cam2world = [np.loadtxt(os.path.join(model_path, 'aligned_cam2world', f"{idx:0>3d}.txt")) for idx in range(40)]
                # H = pred_cam2world[0] @ np.linalg.inv(gt_cam2world[0])
                # gt_cam2world = [H @ cam for cam in gt_cam2world]
                # H = gt_cam2world[prev*2]
                # for g in geometries[0]:
                #     g.transform(H)
                # H = pred_cam2world[prev*2]
                # for g in geometries[1]:
                #     g.transform(H)

                for i in range(len(pred_prev_frame)):
                    cam2world = pred_prev_frame[i]
                    ret = create_camera_frustum(cam2world, intrinsics[i%len(cams)], img_size=(height, width), frustum_length=0.1, color=[0.106, 0.620, 0.467])
                    geometries[1].extend(ret)
                for i in range(len(pred_this_frame)):
                    cam2world = pred_this_frame[i]
                    ret = create_camera_frustum(cam2world, intrinsics[i%len(cams)], img_size=(height, width), frustum_length=0.1, color=[0.914, 0.541, 0.765])
                    geometries[1].extend(ret)
                # for cam2world in gt_cam2world:
                #     ret = create_camera_frustum(cam2world, intrinsics[0], img_size=(height, width), frustum_length=0.1, color=[0.914, 0.541, 0.765])
                #     geometries[0].extend(ret)
                #     geometries[1].extend(ret)

                for geoms in geometries:
                    vis.clear_geometries()
                    for g in geoms:
                        vis.add_geometry(g, reset_bounding_box=first)
                    vis.poll_events()
                    vis.update_renderer()
                    first = False  # 之后不再重置视角

                    # —— 阻塞等待（按 N 继续 / Q 退出），不需要可删 —— #
                    state["step"] = False
                    print(f"[{scene_dir}-{model_dir}-{file}] Press 'N' for next, 'Q' to quit.")
                    while not state["step"] and not state["quit"]:
                        vis.poll_events()
                        vis.update_renderer()
                    if state["quit"]:
                        visualize = False
                        vis.destroy_window()
                        raise SystemExit(0)