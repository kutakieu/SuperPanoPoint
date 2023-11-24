from typing import List, Optional

import torch
from einops import rearrange
from torchvision.transforms.v2 import (Compose, Normalize,
                                       RandomHorizontalFlip, ToTensor)

from . import BaseDataset, DataSample
from .data_synth import generate_perspective_sample


class SyntheticDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], gen_onthefly: bool, num_samples: Optional[int]=None, flip=False, **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.gen_onthefly = gen_onthefly
        self.num_samples = num_samples
        self.img_w = kwargs.get("img_w", 512)
        self.img_h = kwargs.get("img_h", 512)
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
        if self.gen_onthefly:
            return self.num_samples
        return len(self.data_samples)

    def __getitem__(self, index: int):
        if self.gen_onthefly:
            synth_data = generate_perspective_sample(self.img_w, self.img_h)
            img = synth_data.synth_img
            points = synth_data.points_as_img()
        else:
            sample = self.data_samples[index]
            img = sample.load_img()
            points = sample.load_points()
        img = self.img_transform(img)
        points = torch.Tensor(points).permute(2, 0, 1) # (h, w, c) => (c h w)
        if self.common_transform:
            img, points = self.apply_common_transform(img, points)
        
        return img, self.rearrange_points_img(points)

    def __repr__(self):
        return f"SyntheticDataset: {len(self)} samples"
    
    def apply_common_transform(self, img: torch.Tensor, points: torch.Tensor):
        img_channels = img.shape[0]
        stacked = torch.vstack([img, points])
        stacked_transformed = self.common_transform(stacked)
        return stacked_transformed[:img_channels], stacked_transformed[img_channels:]
    
    def rearrange_points_img(self, points: torch.Tensor):
        c, h, w = points.shape
        points = rearrange(points[0], "(hc ch1) (wc ch2) -> hc wc ch1 ch2", hc=h//8, wc=w//8, ch1=8, ch2=8)  # (1, h, w) => (h/8, w/8, 8, 8)
        points = rearrange(points, "h w ch1 ch2 -> h w (ch1 ch2)", ch1=8, ch2=8)  # (h/8, w/8, 8, 8) => (h/8, w/8, 64)
        is_point_not_exist = (points.sum(dim=2) == 0).unsqueeze(-1)  # (h/8, w/8)
        points = torch.concat([points, is_point_not_exist], dim=2).permute(2, 0, 1)  # (h/8, w/8, 65) => (65, h/8, w/8)
        return torch.argmax(points, dim=0, keepdim=False)  # (h/8, w/8)
