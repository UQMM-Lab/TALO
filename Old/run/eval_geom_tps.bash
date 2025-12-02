interframe_solver_choice=tps+v0.05e0.005
tps_settings=(
    v0.1e0.005
)
dataset_ls=(
    nuscenes
    waymo_less
)
model_ls=(
    Pi3
    VGGT
)
conf_threshold_ls=(
    50
    # 60
)
for dataset in ${dataset_ls[@]}; do
    if [ $dataset == "nuscenes" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
        scene_ls=(
            scene-0003
            scene-0012
            scene-0013
            scene-0036 
            scene-0039 #
            scene-0092 #
            scene-0094 #

        )
    fi
    if [ $dataset == "waymo_less" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
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
    for scene in ${scene_ls[@]}; do
        for conf_threshold in ${conf_threshold_ls[@]}; do
            for model in ${model_ls[@]}; do
                for setting in ${tps_settings[@]}; do
                    save_dir=$model+$conf_threshold+$interframe_solver_choice\($setting\)
                    echo ------------$scene/$save_dir
                    data_folder=$data_root/$scene

                    # python main.py --data_folder $data_folder --log_path $save_root/$scene/$save_dir --model $model --conf_threshold $conf_threshold --interframe_solver_choice $interframe_solver_choice --cp_setting $setting  
                    # python fit_tps.py $save_root/$scene/$save_dir
                    # python eval_vis_pcd_traj.py --GT $data_folder --pred $save_root/$scene/$save_dir --eval_pcd # --vis_eval_pcd #--vis --vis_eval_traj
                done
            done
        done
    done
    python summary_geom.py $save_root
done
