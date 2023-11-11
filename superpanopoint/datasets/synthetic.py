from typing import List

import torch
from torchvision.transforms.v2 import (Compose, Normalize,
                                       RandomHorizontalFlip, ToDtype, ToTensor)

from . import BaseDataset, DataSample


class SingleImageDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], flip=True, **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.flip = flip
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.449], std=[0.226])
        ])
        self.points_transform = Compose([
            ToDtype(dtype=torch.float32, scale=False),
        ])
        common_transform_list = []
        if flip:
            common_transform_list.append(RandomHorizontalFlip())
        self.common_transform = Compose(common_transform_list) if common_transform_list else None

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        img = self.img_transform(sample.load_img())
        points = self.points_transform(torch.Tensor(sample.load_points()))
        if self.common_transform:
            img, points = self.apply_common_transform(img, points)
        return img, points

    def __repr__(self):
        return f"PointDataset: {len(self)} samples"
    
    def apply_common_transform(self, img: torch.Tensor, points: torch.Tensor):
        img_channels = img.shape[0]
        stacked = torch.vstack([img, points])
        stacked_transformed = self.common_transform(stacked)
        return stacked_transformed[:img_channels], stacked_transformed[img_channels:]

