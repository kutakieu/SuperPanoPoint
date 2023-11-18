from typing import List

import torch
from einops import rearrange
from torchvision.transforms.v2 import (Compose, Normalize,
                                       RandomHorizontalFlip, ToTensor)

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
        common_transform_list = []
        if flip:
            common_transform_list.append(RandomHorizontalFlip())
        self.common_transform = Compose(common_transform_list) if common_transform_list else None

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        img = self.img_transform(sample.load_img())
        points = torch.Tensor(sample.load_points()).permute(2, 0, 1) # (h, w, c) => (c h w)
        if self.common_transform:
            img, points = self.apply_common_transform(img, points)
        
        return img, self.rearrange_points_img(points)

    def __repr__(self):
        return f"PointDataset: {len(self)} samples"
    
    def apply_common_transform(self, img: torch.Tensor, points: torch.Tensor):
        img_channels = img.shape[0]
        stacked = torch.vstack([img, points])
        stacked_transformed = self.common_transform(stacked)
        return stacked_transformed[:img_channels], stacked_transformed[img_channels:]
    
    def rearrange_points_img(self, points: torch.Tensor):
        c, h, w = points.shape
        points = rearrange(points[0], "(hc ch1) (wc ch2) -> hc wc ch1 ch2", hc=h//8, wc=w//8, ch1=8, ch2=8)  # (1, h, w) => (h/8, w/8, 8, 8)
        points = rearrange(points, "h w ch1 ch2 -> h w (ch1 ch2)", ch1=8, ch2=8)  # (h/8, w/8, 8, 8) => (h/8, w/8, 64)
        is_point_exist = (points.sum(dim=2) > 0).unsqueeze(-1)  # (h/8, w/8)
        return torch.concat([points, is_point_exist], dim=2).permute(2, 0, 1)  # (h/8, w/8, 65) => (65, h/8, w/8)
        
