from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from .base_predictor import BasePredictor
from .postprocess import non_maximum_suppression


class UnetPredictor(BasePredictor):
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        super().__init__(cfg, weight_file, device)
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: np.ndarray, pointness_threshold: float=0.5) -> tuple[np.ndarray, np.ndarray]:
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pointness, desc_map = self.net(img_tensor)
            points = self.postprocess_pointness(pointness, pointness_threshold)
            desc = self.postprocess_descriptor(desc_map, points)

        return points, desc
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = self.img_transform(img)
        return img.unsqueeze(0)

    @classmethod
    def postprocess_pointness(cls, pointness: torch.Tensor, pointness_threshold: float, apply_nms=True) -> np.ndarray:
        pointness = pointness.detach().cpu().numpy()[0, 0, :, :]
        h, w = pointness.shape[:2]
        pointness[pointness < pointness_threshold] = 0
        if apply_nms:
            pointness = non_maximum_suppression(pointness, radius=4, strict=True)
        return cls._to_points_array(pointness, pointness_threshold)

    @classmethod
    def postprocess_descriptor(cls, desc: torch.Tensor, points: np.ndarray) -> np.ndarray:
        desc = desc.permute(0, 2, 3, 1).detach().cpu().numpy()  # (bs, h, w, ch)
        return desc[0, points[:, 1], points[:, 0], :]

    @classmethod
    def _to_points_array(cls, img: np.ndarray, pointness_threshold: float) -> np.ndarray:
        coords = np.array(np.where(img > pointness_threshold)).T.astype(float)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        return np.round(coords).astype(int)
    
    def get_desc_map(self, img):
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pointness, desc_map = self.net(img_tensor)
        desc_map = desc_map.permute(0, 2, 3, 1).detach().cpu().numpy()  # (bs, h, w, ch)
        return desc_map[0]
