from random import shuffle
from typing import List

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import ColorJitter, Compose, Normalize, ToTensor

from . import BaseDataset, DataSample
from .data_synth.homographies import (TransformHomography,
                                      generate_random_homography)


class HomographicContrastiveDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], num_contrastive_pairs: int=10, crop_size: int=256, flip: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.num_contrastive_pairs = num_contrastive_pairs
        self.crop_size = crop_size
        self.flip = flip
        self.img_transform = Compose([
            ToTensor(),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_samples)
    
    def __repr__(self):
        return f"PointDataset: {len(self)} samples"

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        img = np.array(sample.load_img(as_gray=False))
        pointness = sample.load_points()[:,:,0]
        if self.crop_size is not None:
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            pointness = cv2.resize(pointness, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        if self.flip:
            img, pointness = self._random_horizontal_flip(img, pointness)
        h, w = img.shape[:2]
        rows, cols = np.where(pointness > 0)
        points_xys = np.hstack([cols[:,np.newaxis], rows[:,np.newaxis]])  # (n, 2)
        homography = generate_random_homography(w, h)
        warped_img, warped_pointness = self.make_homographic_sample(img, pointness, homography)
        warped_points_xys = homography.apply_to_coordinates(points_xys).round().astype(int)

        contrastive_points_idxs, target_points_idxs = [], []
        valid_rows, valid_cols = (homography.make_mask()).nonzero()
        valid_xys = np.hstack([valid_cols[:,np.newaxis], valid_rows[:,np.newaxis]])  # (n, 2)
        idxs = list(range(len(points_xys)))
        shuffle(idxs)
        for i in idxs:
            x, y = points_xys[i]
            xw, yw = warped_points_xys[i]
            if not (0<=xw<w and 0<=yw<h):
                continue
            contrastive_points_idxs.append([w*y+x for x,y in self.generate_pairs(valid_xys, (xw, yw), size=self.num_contrastive_pairs)])
            target_points_idxs.append(w*y+x)
            if len(target_points_idxs) == 40:
                break
        if len(target_points_idxs) < 40:
            return self.__getitem__(np.random.randint(0, len(self)))
        
        return self.img_transform(Image.fromarray(img)), \
            self.img_transform(Image.fromarray(warped_img)), \
            pointness.astype(int), \
            warped_pointness.astype(int), \
            np.array(target_points_idxs), \
            np.array(contrastive_points_idxs)
    
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

    def make_homographic_sample(self, img: np.ndarray, pointness: np.ndarray, homography: TransformHomography):
        warped_img = homography.apply(img)
        warped_points = homography.apply(pointness)
        return warped_img, warped_points

    def generate_pairs(self, valid_xys, target_xy, size: int, dist_threshold: int=8):
        pairs = [target_xy]
        while len(pairs) < size:
            valid_xy = valid_xys[np.random.randint(0, len(valid_xys))]
            if np.linalg.norm(valid_xy - target_xy) < dist_threshold:
                continue
            pairs.append(valid_xy)
        return pairs
