from typing import List

import numpy as np
import torch
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToTensor

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
        self.points_transform = Compose([
            ToDtype(dtype=torch.float32, scale=False),
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
        points = self.points_transform(torch.Tensor(sample.load_points()))
        h, w = img.shape[:2]
        homography = generate_random_homography(w, h)
        warped_img, warped_points = self.make_homographic_sample(sample, homography)
        warped_img = self.img_transform(warped_img)
        warped_points = self.points_transform(torch.Tensor(warped_points))

        correspondence_mask = homography.get_correspondence_mask()

        if self.common_transform:
            img, warped_img, points, warped_points = self.apply_common_transform(img, warped_img, points, warped_points)
        return img, warped_img, points, warped_points, correspondence_mask

    def make_homographic_sample(self, sample: DataSample, homography: TransformHomography):
        img = np.array(sample.load_img())
        points = sample.load_points()
        warped_img = homography.apply(img)
        warped_points = homography.apply(points)
        return warped_img, warped_points
    
    def apply_common_transform(self, img: torch.Tensor, warp_img: torch.Tensor, points: torch.Tensor, warp_points: torch.Tensor):
        img_channels = img.shape[0]
        stacked = torch.vstack([img, warp_img])
        stacked_transformed = self.common_transform(stacked)
        return stacked_transformed[:img_channels], stacked_transformed[img_channels:], points, warp_points

