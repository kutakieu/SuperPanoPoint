from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from .base_predictor import BasePredictor
from .superpoint import postprocess_pointness


class MagicPointPredictor(BasePredictor):
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        super().__init__(cfg, weight_file, device)
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.449], std=[0.226])
        ])

    def __call__(self, img: Union[Image.Image, np.ndarray], return_as_array: bool=False, omit_edge_width: int=4) -> np.ndarray:
        img = np.array(img)
        h,w = img.shape[:2]
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pointness = self.net(img_tensor)
            pointness = postprocess_pointness(pointness)[0]
        pointness = self._omit_points_on_edge(pointness, omit_edge_width=omit_edge_width)

        if return_as_array:
            return self._to_points_array(pointness, w, h)

        if h != pointness.shape[0] or w != pointness.shape[1]:
            pointness = cv2.resize(pointness, (w, h), interpolation=cv2.INTER_NEAREST)
        return pointness

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """convert RGB image to grayscale image and resize to multiple of 8"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        h, w = img.shape[:2]
        if h % 8 != 0 or w % 8 != 0:
            img = cv2.resize(img, (w // 8 * 8, h // 8 * 8))
        img = self.img_transform(img)
        return img.unsqueeze(0)
    
    def calc_prob_map(self, img: Union[Image.Image, np.ndarray], omit_edge_width: int=4) -> float:
        img = np.array(img)
        h,w = img.shape[:2]
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pointness = self.net(img_tensor)
            pointness = postprocess_pointness(pointness, apply_nms=False)[0]
        pointness[:, :omit_edge_width] = 0
        pointness[:, -omit_edge_width:] = 0
        pointness[:omit_edge_width, :] = 0
        pointness[-omit_edge_width:, :] = 0
        return cv2.resize(pointness, (w, h), interpolation=cv2.INTER_CUBIC)
    
    def _omit_points_on_edge(self, pointness: np.ndarray, omit_edge_width: int=4) -> np.ndarray:
        pointness[:, :omit_edge_width] = 0
        pointness[:, -omit_edge_width:] = 0
        pointness[:omit_edge_width, :] = 0
        pointness[-omit_edge_width:, :] = 0
        return pointness

    def _to_points_array(self, img: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        coords = np.array(np.where(img > 0)).T.astype(float)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        coords[:, 0] *= orig_w / w
        coords[:, 1] *= orig_h / h
        return np.round(coords).astype(int)
