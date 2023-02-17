import vtk
import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
import torch.nn.functional as F

# 从3D图像中 重采样 2D图像
def resample_pic_from_volume(volume, origin, plane_X, plane_Y, plane_len):
    X, Y = np.meshgrid(range(plane_len), range(plane_len))
    X_new = np.stack([X, X, X], axis=-1)
    Y_new = np.stack([Y, Y, Y], axis=-1)
    img_grid = origin + X_new * plane_X + Y_new * plane_Y

    volume = volume.copy()
    volume = (volume - volume.min()) * 255 / (volume.max() - volume.min())

    shape_0, shape_1, shape_2 = volume.shape
    coef = 2
    X = (img_grid[..., 0] / (shape_0 - 1) - 0.5) * coef
    Y = (img_grid[..., 1] / (shape_1 - 1) - 0.5) * coef
    Z = (img_grid[..., 2] / (shape_2 - 1) - 0.5) * coef
    img_grid = np.stack([Z, Y, X], axis=-1)

    volume = torch.tensor(volume, dtype=torch.float32)[None, None]
    img_grid = torch.tensor(img_grid, dtype=torch.float32)[None, None]
    ten_resample = F.grid_sample(volume, img_grid)
    # 重采样后的2值图像
    pic = np.array(ten_resample[0, 0, 0])
    pic = pic.astype(np.uint8)
    return pic
