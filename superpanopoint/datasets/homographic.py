from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from superpanopoint.models.predictor import MagicPointPredictor

from . import BaseDataset, DataSample
from .data_synth.homographies import (TransformHomography,
                                      generate_random_homography)


class HomographicDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], point_detector: Optional[MagicPointPredictor], crop_size: int=256, flip: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.point_detector = point_detector
        self.crop_size = crop_size
        self.flip = flip
        self.img_transform = Compose([
            ToTensor(),
            # Normalize(mean=[0.449], std=[0.226]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_samples)
    
    def __repr__(self):
        return f"PointDataset: {len(self)} samples"

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        img = np.array(sample.load_img(as_gray=False))
        if sample.points_file is None:
            if self.point_detector is None:
                raise ValueError("point_detector must be provided if points_file is None")
            points, desc = self.point_detector(img, return_as_array=False)
            points = points[:, :, np.newaxis]
        else:
            points = sample.load_points()
        if self.crop_size is not None:
            img, points = self._random_crop(img, points, self.crop_size)
        if self.flip:
            img, points = self._random_horizontal_flip(img, points)
        h, w = img.shape[:2]
        homography = generate_random_homography(w, h)
        warped_img, warped_points = self.make_homographic_sample(img, points, homography)

        warped_img = self.img_transform(Image.fromarray(warped_img))
        warped_points = torch.Tensor(warped_points).permute(2, 0, 1) # (h, w, c) => (c h w)
        
        img = self.img_transform(Image.fromarray(img))
        points = torch.Tensor(points).permute(2, 0, 1) # (h, w, c) => (c h w)

        correspondence_mask = homography.get_correspondence_mask()

        return img, self.rearrange_points_img(points), warped_img, self.rearrange_points_img(warped_points), torch.Tensor(correspondence_mask)
    
    def _random_crop(self, img: np.ndarray, points: np.ndarray, crop_size: int):
        h, w = img.shape[:2]
        x = np.random.randint(0, w - crop_size)
        y = np.random.randint(0, h - crop_size)
        img = img[y:y+crop_size, x:x+crop_size]
        points = points[y:y+crop_size, x:x+crop_size]
        return img.copy(), points.copy()
    
    def _random_horizontal_flip(self, img: np.ndarray, points: np.ndarray):
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
            points = np.flip(points, axis=1)
        return img.copy(), points.copy()

    def make_homographic_sample(self, img: np.ndarray, points: np.ndarray, homography: TransformHomography):
        warped_img = homography.apply(img)
        warped_points = homography.apply(points)
        if len(warped_points.shape) == 2:
            warped_points = warped_points[:, :, np.newaxis]
        return warped_img, warped_points

    def rearrange_points_img(self, points: torch.Tensor):
        c, h, w = points.shape
        points = rearrange(points[0], "(hc ch1) (wc ch2) -> hc wc ch1 ch2", hc=h//8, wc=w//8, ch1=8, ch2=8)  # (1, h, w) => (h/8, w/8, 8, 8)
        points = rearrange(points, "h w ch1 ch2 -> h w (ch1 ch2)", ch1=8, ch2=8)  # (h/8, w/8, 8, 8) => (h/8, w/8, 64)
        is_point_not_exist = (points.sum(dim=2) == 0).unsqueeze(-1)  # (h/8, w/8)
        points = torch.concat([points, is_point_not_exist], dim=2).permute(2, 0, 1)  # (h/8, w/8, 65) => (65, h/8, w/8)
        return torch.argmax(points, dim=0, keepdim=False)  # (h/8, w/8)
