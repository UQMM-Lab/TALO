import os
import glob
import argparse
import random
import shutil

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver




import ipdb
np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--data_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--log_results", default=True, action="store_true", help="save txt file with results")
parser.add_argument("--log_path", type=str, default="poses", help="Path to save the log file")
parser.add_argument("--port", type=int, default=8080, help="Port for the viewer")

parser.add_argument("--conf_threshold", type=int, default=50, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--interframe_solver_choice", type=str, choices=["pose", "pose+tps+v0.05e0.005", "sim3+tps+v0.05e0.005", 'sim3', 'sl4'], required=True)
parser.add_argument("--model", type=str, default="VGGT", choices=["VGGT", "Pi3", "MapAnything"], help="Choice of 3D foundation model")
parser.add_argument("--cam_num", type=int, default=6, help="Number of cameras used in the multi-camera rig")


parser.add_argument("--max_loops", type=int, default=0, help="Maximum number of loop closures per submap")

parser.add_argument("--vis_tps", action="store_true", help="Visualize the TPS allignment")
parser.add_argument("--allow_nonvalid", default=False, action="store_true", help="Allow adding submaps even if no valid points are found")
parser.add_argument("--backward_control_points", default=False, action="store_true", help="Use backward control points for SL4")
# parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for downsampling the point cloud for visualization and ICP")
# parser.add_argument("--max_project_error", type=float, default=None, help="Maximum reprojection error (in depth) for a point to be considered valid during inter-frame alignment")
parser.add_argument("--filter_method", type=str, default="gaussian", choices=["mean", "gaussian", "robust", None, 'None'], help="Method for robust mean filtering of control points")
parser.add_argument("--filter_k", type=int, default=32, help="Number of nearest neighbors to use for robust mean filtering of control points")
parser.add_argument("--robust_mean_method", type=str, default='mad', choices=["mad", "std", None, 'None'], help="Method for robust mean filtering of control points")
parser.add_argument("--submap_size", type=int, default=2, help="Number of frames per submap")


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # use all as overlapping
    
    if 'nuscenes' in args.data_folder.lower():
        cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]
        # cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    elif 'waymo' in args.data_folder.lower():
        cam_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    else:
        raise
    if args.cam_num < len(cam_names):
        cam_names = cam_names[:args.cam_num]
    zero_pad = 3
    submap_size = overlap_size= args.submap_size
    
    ref_cam_id=0
    cam_num = len(cam_names)

    if 'tps' in args.interframe_solver_choice:
        cp_setting = args.interframe_solver_choice.split('+')[-1]
        voxel_size = float(cp_setting.split('v')[1].split('e')[0])
        max_project_error_rate = float(cp_setting.split('e')[1])
        print(f"Using TPS inter-frame solver, voxel size: {voxel_size}, max projection error rate: {max_project_error_rate}")
    else:
        voxel_size = None
        max_project_error_rate = None
    max_loops = args.max_loops * cam_num  
    solver = Solver(
        model=args.model,
        init_conf_threshold=args.conf_threshold,
        visualize_global_map=args.vis_map,
        interframe_solver_choice=args.interframe_solver_choice,
        gradio_mode=False,
        port=args.port,
        save_root=args.log_path,
        overlap_size=overlap_size,
        submap_size=submap_size,
        cam_names=cam_names,
        ref_cam_id=ref_cam_id,
        ref_cam_name=cam_names[ref_cam_id],
        vis_tps=args.vis_tps,
        allow_nonvalid=args.allow_nonvalid,
        voxel_size_ratio=voxel_size,
        projection_error_ratio=max_project_error_rate,
        filter_method=args.filter_method,
        filter_k=args.filter_k,
        robust_mean_method=args.robust_mean_method,
        backward_control_points=args.backward_control_points,
    )
    
    print("Initializing and loading 3D foundation model...")

    if args.model == "VGGT":
        from vggt.models.vggt import VGGT
        # model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model = model.to(device).eval()
        
    elif args.model == "Pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    elif args.model == "MapAnything":
        from mapanything.models import MapAnything
        model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()


    timestep_num = len(os.listdir(os.path.join(args.data_folder, f"image/{cam_names[0]}")))

    frame_ids = []
    print(f"Found {timestep_num} timesteps")
    shutil.rmtree(args.log_path, ignore_errors=True)
    os.makedirs(args.log_path, exist_ok=True)

    for timestep in range(timestep_num):
        frame_ids += [f"{cam_name}/{timestep:0>{zero_pad}d}" for cam_name in cam_names]  # assuming each camera has the same number of images
        if len(frame_ids) < submap_size * cam_num + overlap_size * cam_num and timestep < timestep_num - 1:
            continue
        # print(f"Processing frame {timestep}, total frames in current submap: {len(frame_ids)}")
        image_names_subset = [f"{args.data_folder}/image/{frame_id}.jpg" for frame_id in frame_ids]  # assuming each camera has the same number of images

        predictions = solver.run_predictions(image_names_subset, model, max_loops=max_loops, frame_ids=frame_ids)

        solver.add_points(predictions)

        solver.graph.optimize()
        solver.map.update_submap_homographies(solver.graph)

        loop_closure_detected = len(predictions["detected_loops"]) > 0

        if args.vis_map:
            solver.update_all_submap_vis()
            # if loop_closure_detected:
            # else:
            #     solver.update_latest_submap_vis()
            
        # Reset for next submap.
        frame_ids = frame_ids[-overlap_size*cam_num:]
        # print(image_names_subset)

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    

    if "tps" in args.interframe_solver_choice:
        solver.fit_tps_global()

    solver.cal_rig()
    
    if args.vis_map:
        solver.update_all_submap_vis(True)

    solver.write_to_each_camera()

    # Log the full point cloud as one file, used for visualization.
    # solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))

    # Log the dense point cloud for each submap.
    solver.save_framewise_pointclouds()


if __name__ == "__main__":
    main()
