import json
import os
import sys
import glob
import ipdb
import numpy as np

root = sys.argv[1]
results_all = {}
models = ['VGGT', 'Pi3', "MapAnything"]
methods = ["sim3", "sl4", "tps"]
for scene_dir in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(scene_dir):
        continue
    scene_name = os.path.basename(scene_dir)
    results_scene = {}
    for model_dir in sorted(os.listdir(scene_dir)):
        is_in = False
        for model in models:
            if model in model_dir:
                is_in = True
                break
        if not is_in:
            continue
        is_in = False
        for method in methods:
            if method in model_dir:
                is_in = True
                break
        if not is_in:
            continue
        # print(f"Processing {scene_name} - {model_dir}")
        model_path = os.path.join(scene_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        metrics_path = os.path.join(model_path, "eval_metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found: {metrics_path}")
            continue
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        model_name, confi, *config = os.path.basename(model_path).split("+")
        if confi not in results_scene:
            results_scene[confi] = {}
        if model_name not in results_scene[confi]:
            results_scene[confi][model_name] = {}
        results_scene[confi][model_name]["+".join(config)] = metrics['trajectory']

    results_all[scene_name] = results_scene

metrics = ["ATE_RMSE", "RTE_RMSE", "RRE_RMSE"]

# {method: {metric: [values...]}}
stats = {model: {method: {metric: [] for metric in metrics} for method in methods} for model in models}
for scene_name, scene_data in results_all.items():
    for confi, confi_data in scene_data.items():
        for model_name, model_data in confi_data.items():
            for method_name, method_data in model_data.items():
                if method_name not in methods:
                    continue
                for metric in metrics:
                    if metric in method_data:
                        stats[model_name][method_name][metric].append(method_data[metric])


# print results
results_all['Avg'] = {
    model: {
        method: {
            metric: float(np.mean(stats[model][method][metric])) if len(stats[model][method][metric]) > 0 else None
            for metric in metrics
        }
        for method in methods
    }
    for model in models
}


output_path = os.path.join(root, "metrics_traj.json")
with open(output_path, "w") as f:
    json.dump(results_all, f, indent=4)
print(f"Saved summarized metrics to {output_path}")