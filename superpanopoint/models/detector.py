import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from . import model_factory
from .decoders.superpoint import postprocess_descriptor, postprocess_pointness


class Predictor:
    def __init__(self, cfg, weight_file, device="cpu") -> None:
        self.device = device
        self.net = model_factory(cfg)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(weight_file)["state_dict"])
        self.net.eval()
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.449], std=[0.226])
        ])

    def __call__(self, img: np.ndarray, return_as_array: bool=False) -> tuple[np.ndarray, np.ndarray]:
        h,w = img.shape[:2]
        img_tensor = self._preprocess(img).to(self.device)
        with torch.no_grad():
            pred_pointness, desc = self.net(img_tensor)
            pred_pointness = self._postprocess(pred_pointness)
            desc = postprocess_descriptor(desc)[0]

        if return_as_array:
            return self._to_points_array(pred_pointness, w, h)
        if h != pred_pointness.shape[0] or w != pred_pointness.shape[1]:
            pred_pointness = cv2.resize(pred_pointness, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred_pointness, desc
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        if h % 8 != 0 or w % 8 != 0:
            img = cv2.resize(img, (w // 8 * 8, h // 8 * 8))
        img = self.img_transform(img)
        return img.unsqueeze(0)
    
    def _postprocess(self, pred_pointness: torch.Tensor) -> np.ndarray:
        """input: (1, 65, H/8, W/8), output: (H, W)"""
        return postprocess_pointness(pred_pointness)[0]
    
    def _to_points_array(self, img: np.ndarray, orig_w, orig_h) -> np.ndarray:
        h, w = img.shape[:2]
        coords = np.array(np.where(img > 0)).T.astype(float)
        coords[:, 0] *= orig_h / h
        coords[:, 1] *= orig_w / w
        return np.round(coords).astype(int)
