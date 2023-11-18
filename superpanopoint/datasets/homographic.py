from typing import List

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from . import BaseDataset, DataSample
from .data_synth.homographies import (TransformHomography,
                                      generate_random_homography)


class DoubleImageDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.449], std=[0.226])
        ])
        common_transform_list = []
        self.common_transform = Compose(common_transform_list) if common_transform_list else None

    def __len__(self):
        return len(self.data_samples)
    
    def __repr__(self):
        return f"PointDataset: {len(self)} samples"

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        img = self.img_transform(sample.load_img())
        points = torch.Tensor(sample.load_points()).permute(2, 0, 1) # (h, w, c) => (c h w)
        c, h, w = img.shape
        homography = generate_random_homography(w, h)
        warped_img, warped_points = self.make_homographic_sample(sample, homography)
        warped_img = self.img_transform(warped_img)
        warped_points = torch.Tensor(warped_points).permute(2, 0, 1) # (h, w, c) => (c h w)

        correspondence_mask = homography.get_correspondence_mask()

        if self.common_transform:
            img, warped_img, points, warped_points = self.apply_common_transform(img, warped_img, points, warped_points)
        return img, warped_img, self.rearrange_points_img(points), self.rearrange_points_img(warped_points), correspondence_mask

    def make_homographic_sample(self, sample: DataSample, homography: TransformHomography):
        img = np.array(sample.load_img())
        points = sample.load_points()
        warped_img = homography.apply(img)
        warped_points = homography.apply(points)
        if len(warped_points.shape) == 2:
            warped_points = warped_points[:, :, np.newaxis]
        return warped_img, warped_points
    
    def apply_common_transform(self, img: torch.Tensor, warp_img: torch.Tensor, points: torch.Tensor, warp_points: torch.Tensor):
        img_channels = img.shape[0]
        stacked = torch.vstack([img, warp_img])
        stacked_transformed = self.common_transform(stacked)
        return stacked_transformed[:img_channels], stacked_transformed[img_channels:], points, warp_points

    def rearrange_points_img(self, points: torch.Tensor):
        c, h, w = points.shape
        points = rearrange(points[0], "(hc ch1) (wc ch2) -> hc wc ch1 ch2", hc=h//8, wc=w//8, ch1=8, ch2=8)  # (1, h, w) => (h/8, w/8, 8, 8)
        points = rearrange(points, "h w ch1 ch2 -> h w (ch1 ch2)", ch1=8, ch2=8)  # (h/8, w/8, 8, 8) => (h/8, w/8, 64)
        is_point_exist = (points.sum(dim=2) > 0).unsqueeze(-1)  # (h/8, w/8)
        return torch.concat([points, is_point_exist], dim=2).permute(2, 0, 1)  # (h/8, w/8, 65) => (65, h/8, w/8)
