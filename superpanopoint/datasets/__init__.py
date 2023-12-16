import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from superpanopoint import Settings

xs = np.linspace(-1, 1, 9)
ys = np.linspace(-1, 1, 9)
xs, ys = np.meshgrid(xs, ys)
sigma_pow2 = 0.4 ** 2
gaussian_kernel = np.exp(-(xs**2 + ys**2)/(2*sigma_pow2)) / (2 * np.pi * sigma_pow2)
gaussian_kernel /= np.max(gaussian_kernel)
kh, kw = gaussian_kernel.shape

@dataclass
class DataSample:
    img_file: Path
    points_file: Optional[Path]
    img_width: Optional[int] = None
    img_height: Optional[int] = None

    def load_img(self, as_gray=True):
        if as_gray:
            img = Image.open(self.img_file).convert("L")
        else:
            img = Image.open(self.img_file)
        self.img_width, self.img_height = img.size
        return img
    
    def load_points(self, as_probmap: bool=False)->Optional[np.ndarray]:
        """
        Returns:
            Optional[np.ndarray]: binary points image as a numpy array shape: (h, w, 1)
        """
        if self.points_file is None:
            return None
        if self.img_width is None or self.img_height is None:
            self.load_img()

        with open(self.points_file, "r") as f:
            d = json.load(f)
        point_img = np.zeros((self.img_height, self.img_width, 1), dtype=float)
        for point in d[Settings().points_key]:
            if as_probmap and 0<=point["y"]-kh and point["y"]+kh<self.img_height and 0<=point["x"]-kw and point["x"]+kw<self.img_width:
                point_img[point["y"]-kh//2:point["y"]+kh//2+1, point["x"]-kw//2:point["x"]+kw//2+1, 0] = gaussian_kernel
            else:
                point_img[point["y"], point["x"], :] = 1

        return point_img
    

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
