import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data_synth import IMG_HEIGHT_KEY, IMG_WIDTH_KEY, POINTS_KEY


@dataclass
class DataSample:
    img_file: Path
    points_file: Path

    def load_img(self):
        return Image.open(self.img_file)
    
    def load_points(self):
        with open(self.points_file, "r") as f:
            d = json.load(f)
        point_img = np.zeros((d[IMG_HEIGHT_KEY], d[IMG_WIDTH_KEY], 1), dtype=float)
        for point in d[POINTS_KEY]:
            point_img[point["y"], point["x"], :] = 1
        return point_img
    

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
