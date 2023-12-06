import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from superpanopoint import Settings


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
    
    def load_points(self)->Optional[np.ndarray]:
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
            point_img[point["y"], point["x"], :] = 1
        return point_img
    

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
