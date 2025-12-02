dataset_ls=(
    waymo_less
    # nuscenes
)
model_ls=(
    VGGT
    # Pi3
    # MapAnything
)
interframe_solver_choice_ls=(
    # pose
    # pose+scale
    # sim3
    # sl4
    pose+tps+v0.05e0.005
)
conf_threshold_ls=(
    60
)
for dataset in ${dataset_ls[@]}; do
    if [ $dataset == "nuscenes" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
        scene_ls=(
            scene-0003
            scene-0012
            scene-0013
            scene-0039 #
            scene-0094 #

        )
    fi
    if [ $dataset == "waymo_less" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
        scene_ls=(
            3156155872654629090
            3461811179177118163
            4058410353286511411 # perfect
            6104545334635651714
            16345319168590318167

        )
    fi
    for scene in ${scene_ls[@]}; do
        for conf_threshold in ${conf_threshold_ls[@]}; do
            for model in ${model_ls[@]}; do
                for interframe_solver_choice in ${interframe_solver_choice_ls[@]}; do
                    echo ------------$scene $model $conf_threshold $interframe_solver_choice
                    data_folder=$data_root/$scene
                    save_dir=$model+$conf_threshold+$interframe_solver_choice

                    # python main.py --data_folder $data_folder --log_path $save_root/$scene/$save_dir --model $model --conf_threshold $conf_threshold --interframe_solver_choice $interframe_solver_choice # --vis_map  --port 8089 #--vis_tps #

                    python fit_tps.py $save_root/$scene/$save_dir

                    # python eval_vis_pcd_traj.py --GT $data_folder --pred $save_root/$scene/$save_dir --eval_traj 

                done
            done
        done
    done
done