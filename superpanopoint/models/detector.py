import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from . import model_factory
from .decoders.superpoint import postprocess_pointness


class PointDetector:
    def __init__(self, cfg, weight_file) -> None:
        self.net = model_factory(cfg)
        self.net.load_state_dict(torch.load(weight_file)["state_dict"])
        self.net.eval()
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.449], std=[0.226])
        ])

    def __call__(self, img: np.ndarray, return_as_array: bool=False) -> np.ndarray:
        img_tensor = self._preprocess(img)
        with torch.no_grad():
            pred_pointness, _ = self.net(img_tensor)
            pred_pointness = self._postprocess(pred_pointness)

        if return_as_array:
            return self._to_points_array(pred_pointness)
        
        return pred_pointness
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(img)
        return img.unsqueeze(0)
    
    def _postprocess(self, pred_pointness: torch.Tensor) -> np.ndarray:
        """input: (1, 65, H/8, W/8), output: (H, W)"""
        return postprocess_pointness(pred_pointness)[0]
    
    def _to_points_array(self, img: np.ndarray) -> np.ndarray:
        return np.array(np.where(img > 0)).T
        
