from typing import List

import numpy as np

from superpanopoint import Settings
from superpanopoint.models.detector import Predictor
from superpanopoint.models.utils.postprocess import non_maximum_suppression

from .homographies import TransformHomography, generate_random_homography


class HomographicAdapter:
    def __init__(self, img_w: int, img_h: int, detector: Predictor, num_homographyies: int=100) -> None:
        self.homographies: List[TransformHomography] = [generate_random_homography(img_w, img_h) for _ in range(num_homographyies)]
        self.detector = detector

    def generate_pseudo_labels(self, img: np.ndarray, asjson: bool=True) -> np.ndarray:
        pseudo_label = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        for homography in self.homographies:
            warped_img = homography.apply(img)
            point_img, desc = self.detector(warped_img)
            pseudo_label += homography.apply_inverse(point_img)
        pseudo_label = np.heaviside(pseudo_label-0.5, 1)

        pseudo_label = non_maximum_suppression(pseudo_label)
        if asjson:
            return self.convert_to_json(pseudo_label)
        return pseudo_label
    
    def convert_to_json(self, points_img: np.ndarray):
        rows, cols = np.where(points_img > 0)
        rows = rows.tolist()
        cols = cols.tolist()
        return {
            Settings().img_width_key: points_img.shape[1],
            Settings().img_height_key: points_img.shape[0],
            Settings().points_key: [{"x": col, "y": row} for row, col in zip(rows, cols)]
        }
