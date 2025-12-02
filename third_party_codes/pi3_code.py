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

'''
The following codes are adapted from Pi3 (https://github.com/yyfz/Pi3/blob/main/pi3/utils/basic.py)
'''
def load_images_as_tensor(image_paths, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    # print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    # print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

'''The following codes are modified from MoGe (https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py)'''

from typing import *
import math

import numpy as np
import torch
import torch.nn.functional as F
from typing import *
import math
import os


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def solve_focal_no_shift(uv: np.ndarray, xyz: np.ndarray, z_eps: float = 1e-6) -> float:
    """
    assume shift=0, solve the least squares min || f * (xy/z) - uv ||_2 in closed form.
    uv:  (N, 2)   target normalized pixel coordinates (consistent with normalized_view_plane_uv)
    xyz: (N, 3)   points in camera coordinate system
    Returns: scalar focal (dimensionless relative to half-diagonal, which can be converted to fx, fy by your code)
    """
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z  = xyz[..., 2].reshape(-1)

    # Filter out invalid/near-zero z to prevent division by zero and outliers
    m = np.isfinite(z) & (np.abs(z) > z_eps)
    if m.sum() < 2:
        return 1.0

    a = xy[m] / z[m, None]          # (M,2)  = xy/z
    b = uv[m]                        # (M,2)

    num = (a * b).sum()             # <a,b>
    den = (a * a).sum()             # <a,a>
    if den <= 0 or not np.isfinite(num / den):
        return 1.0
    return float(num / den)


import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def recover_focal_no_shift(points: torch.Tensor,
                           mask: torch.Tensor = None,
                           downsample_size: Tuple[int, int] = (64, 64)):
    """
    assume that points are already in the camera coordinate system (shift=0), only estimate focal.
    Returns focal: (...), does not return shift.
    """
    shape = points.shape
    H, W = points.shape[-3], points.shape[-2]

    pts = points.reshape(-1, *shape[-3:])                               # (B, H, W, 3)
    msk = None if mask is None else mask.reshape(-1, *shape[-3:-1])     # (B, H, W)

    pts_lr = F.interpolate(pts.permute(0,3,1,2), downsample_size, mode='nearest').permute(0,2,3,1)  # (B,h,w,3)

    uv = normalized_view_plane_uv(W, H, dtype=points.dtype, device=points.device)                    # (H,W,2)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0,3,1,2), downsample_size, mode='nearest').squeeze(0).permute(1,2,0)  # (h,w,2)

    msk_lr = None if msk is None else (F.interpolate(msk.to(torch.float32).unsqueeze(1),
                                                     downsample_size, mode='nearest').squeeze(1) > 0)

    uv_np     = uv_lr.cpu().numpy()
    pts_np    = pts_lr.detach().cpu().numpy()
    msk_np    = None if msk_lr is None else msk_lr.cpu().numpy()

    focals = []
    for i in range(pts_np.shape[0]):
        if msk_np is None:
            uv_i  = uv_np.reshape(-1, 2)
            xyz_i = pts_np[i].reshape(-1, 3)
        else:
            m = msk_np[i].reshape(-1)
            uv_i  = uv_np.reshape(-1, 2)[m]
            xyz_i = pts_np[i].reshape(-1, 3)[m]

        if uv_i.shape[0] < 2:
            focals.append(1.0); continue

        f_i = solve_focal_no_shift(uv_i, xyz_i)   # ★ only estimate focal, assume shift=0
        focals.append(f_i)

    focal = torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    return focal
