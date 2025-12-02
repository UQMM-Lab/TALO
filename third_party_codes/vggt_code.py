import ipdb
import torch
import math
from PIL import Image
from torchvision import transforms

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tqdm
import trimesh
import gradio as gr
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import cv2
import os
import requests
import pathlib

'''The following codes are adapted from VGGT (https://github.com/facebookresearch/vggt)'''

import onnxruntime
def apply_sky_segmentation(image_files, target_size) -> np.ndarray:
    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []

    for i, image_path in enumerate(image_files):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        path = pathlib.Path(image_path)
        mask_filepath = path.parents[2] / "sky_masks" / path.parents[0].name / image_name.replace(".jpg", ".png")

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
            cv2.imwrite(mask_filepath, sky_mask)
        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != target_size[0] or sky_mask.shape[1] != target_size[1]:
            sky_mask = cv2.resize(sky_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)


        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # # Apply sky mask to confidence scores
    non_sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)

    return non_sky_mask_binary

def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask

def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result

def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

    device, dtype = depth.device, depth.dtype
    height, width = depth.shape[-2:]
    radius = kernel_size // 2

    duv = torch.stack(torch.meshgrid(torch.linspace(-radius / width, radius / width, kernel_size, device=device, dtype=dtype), torch.linspace(-radius / height, radius / height, kernel_size, device=device, dtype=dtype), indexing='xy'), dim=-1).to(dtype=dtype, device=device)

    log_depth = depth.clamp_min_(eps).log()
    log_depth_diff = utils3d.torch.sliding_window_2d(log_depth, window_size=kernel_size, stride=1, dim=(-2, -1)) - log_depth[..., radius:-radius, radius:-radius, None, None] 
    
    weight = torch.exp(-(log_depth_diff / duv.norm(dim=-1).clamp_min_(eps) / 10).square())
    tot_weight = weight.sum(dim=(-2, -1)).clamp_min_(eps)

    uv = utils3d.torch.image_uv(height=height, width=width, device=device, dtype=dtype)
    K_inv = torch.inverse(intrinsics)

    grad = -(normal[..., None, :2] @ K_inv[..., None, None, :2, :2]).squeeze(-2) \
            / (normal[..., None, 2:] + normal[..., None, :2] @ (K_inv[..., None, None, :2, :2] @ uv[..., :, None] + K_inv[..., None, None, :2, 2:])).squeeze(-2)
    laplacian = (weight * ((utils3d.torch.sliding_window_2d(grad, window_size=kernel_size, stride=1, dim=(-3, -2)) + grad[..., radius:-radius, radius:-radius, :, None, None]) * (duv.permute(2, 0, 1) / 2)).sum(dim=-3)).sum(dim=(-2, -1))
    
    laplacian = laplacian.clamp(-0.1, 0.1)
    log_depth_refine = log_depth.clone()

    for _ in range(iterations):
        log_depth_refine[..., radius:-radius, radius:-radius] = 0.1 * log_depth_refine[..., radius:-radius, radius:-radius] + 0.9 * (damp * log_depth[..., radius:-radius, radius:-radius] - laplacian + (weight * utils3d.torch.sliding_window_2d(log_depth_refine, window_size=kernel_size, stride=1, dim=(-2, -1))).sum(dim=(-2, -1))) / (tot_weight + damp) 

    depth_refine = log_depth_refine.exp()

    return depth_refine