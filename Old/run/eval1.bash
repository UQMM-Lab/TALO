dataset=waymo_less
model=VGGT
conf_threshold=60
interframe_solver_choice_ls=(
    # sim3
    # sl4
    pose
    # pose+tps+v0.05e0.005
)
cam=1
if [ $dataset == "waymo_less" ]; then
scene_ls=(
    183829460855609442
    3156155872654629090
    3461811179177118163
    4058410353286511411
    5200186706748209867
    6104545334635651714
    16345319168590318167
)
fi
data_root=Data/$dataset
save_root=ab_camera/Save_$cam/$dataset
for scene in ${scene_ls[@]}; do
    for interframe_solver_choice in ${interframe_solver_choice_ls[@]}; do
        # echo ------------$scene $model $conf_threshold $interframe_solver_choice
        data_folder=$data_root/$scene
        save_dir=$model+$conf_threshold+$interframe_solver_choice

        # python main.py --data_folder $data_folder --log_path $save_root/$scene/$save_dir --model $model --conf_threshold $conf_threshold --interframe_solver_choice $interframe_solver_choice --cam_num $cam
        # python fit_tps.py $save_root/$scene/$save_dir
        # python eval_vis_pcd_traj.py --GT $data_folder --pred $save_root/$scene/$save_dir --eval_pcd --eval_traj
    done
done
python summary_geom.py $save_root
python summary_traj.py $save_root
