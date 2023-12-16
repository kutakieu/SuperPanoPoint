from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from .base_predictor import BasePredictor
from .postprocess import non_maximum_suppression


class SuperPointPredictor(BasePredictor):
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        super().__init__(cfg, weight_file, device)
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h,w = img.shape[:2]
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pointness, desc = self.net(img_tensor)

            pointness = postprocess_pointness(pointness)[0]
            points = self._to_points_array(pointness, w, h)

            pred_desc = postprocess_descriptor(desc, w, h)
        return points, pred_desc[0, points[:, 1], points[:, 0], :]

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """convert RGB image to grayscale image and resize to multiple of 8"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        h, w = img.shape[:2]
        if h % 8 != 0 or w % 8 != 0:
            img = cv2.resize(img, (w // 8 * 8, h // 8 * 8))
        img = self.img_transform(img)
        return img.unsqueeze(0)

    def _to_points_array(self, img: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        coords = np.array(np.where(img > 0)).T.astype(float)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        coords[:, 0] *= orig_w / w
        coords[:, 1] *= orig_h / h
        return np.round(coords).astype(int)

def postprocess_pointness(pointness: torch.Tensor, apply_nms=True, scale=8) -> np.ndarray:
    """
    rearrange pointness tensor to numpy array

    Args:
        - pointness shape: (bs, 65, h_c, w_c).  
            - 65 is 64 + 1 (background)
            - h_c and w_c are height and width of the output of the network
            h_c = h / 8, w_c = w / 8 according to the paper

    Returns:
        - pointness shape: (bs, h, w)
            - h and w are height and width of the original input image
    """
    pointness = pointness.detach().cpu()
    max_idx = torch.argmax(pointness, dim=1, keepdim=True)
    point_mask = torch.FloatTensor(pointness.shape)
    point_mask.zero_().scatter_(1, max_idx, 1)
    point_mask = point_mask[:, :-1, :, :].detach().cpu().numpy()  # (bs, 64, h_c, w_c)
    point_mask = rearrange(point_mask, "b (ch1 ch2) h w -> b h w ch1 ch2", ch1=scale, ch2=scale)  # (bs, h_c, w_c, 8, 8)
    point_mask = rearrange(point_mask, "b h w c1 c2 -> b (h c1) (w c2)")  # (bs, h_c*8, w_c*8)

    if apply_nms:
        pointness = pointness[:, :-1, :, :].detach().cpu().numpy()  # (bs, 64, h_c, w_c)
        pointness = rearrange(pointness, "b (ch1 ch2) h w -> b h w ch1 ch2", ch1=scale, ch2=scale)  # (bs, h_c, w_c, 8, 8)
        pointness = rearrange(pointness, "b h w c1 c2 -> b (h c1) (w c2)")  # (bs, h_c*8, w_c*8)
        pointness = pointness * point_mask  # (bs, h, w)
        for i in range(pointness.shape[0]):
            point_mask[i] = non_maximum_suppression(pointness[i])

    return point_mask  # (bs, h, w)

def postprocess_descriptor(desc: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    desc = F.interpolate(desc, size=(img_h, img_w), mode="bicubic", align_corners=False)
    return desc.permute(0, 2, 3, 1).detach().cpu().numpy()  # (bs, h, w, ch)
