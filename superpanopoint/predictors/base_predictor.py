from abc import abstractmethod
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from superpanopoint.lightning_wrapper import LightningWrapper
from superpanopoint.models import model_factory


class BasePredictor:
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        self.device = device
        self.net = model_factory(cfg)
        self.net.to(self.device)
        if Path(weight_file).suffix == ".ckpt":
            checkpoint = torch.load(weight_file)
            lightning_model = LightningWrapper(cfg)
            lightning_model.load_state_dict(checkpoint['state_dict'])
            self.net = lightning_model.net
        else:
            self.net.load_state_dict(torch.load(weight_file)["state_dict"])
        self.net.eval()
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """convert RGB image to grayscale image and resize to multiple of 8"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        h, w = img.shape[:2]
        if h % 8 != 0 or w % 8 != 0:
            img = cv2.resize(img, (w // 8 * 8, h // 8 * 8))
        img = self.img_transform(img)
        return img.unsqueeze(0)
    
    @abstractmethod
    def _postprocess(self, pred_pointness: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
    
    def _to_points_array(self, img: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        coords = np.array(np.where(img > 0)).T.astype(float)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        coords[:, 0] *= orig_w / w
        coords[:, 1] *= orig_h / h
        return np.round(coords).astype(int)


    
def _to_points_array(img: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    coords = np.array(np.where(img > 0)).T.astype(float)
    coords[:, [0, 1]] = coords[:, [1, 0]]
    coords[:, 0] *= orig_w / w
    coords[:, 1] *= orig_h / h
    return np.round(coords).astype(int)
