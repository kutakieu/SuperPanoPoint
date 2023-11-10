from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataSample:
    img: Path
    points: Path

    def load_img(self):
        return Image.open(self.img)
    
    def load_points(self):
        points = []
        for line in self.points.read_text().splitlines():
            x, y = line.split(",")
            points.append([float(x), float(y)])
        return points

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
