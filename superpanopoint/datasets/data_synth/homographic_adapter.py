from typing import List

import numpy as np

from superpanopoint.models.detector import PointDetector

from .homographies import TransformHomography, generate_random_homography


class HomographicAdapter:
    def __init__(self, img_w: int, img_h: int, detector: PointDetector, num_homographyies: int=100) -> None:
        self.homographies: List[TransformHomography] = [generate_random_homography(img_w, img_h) for _ in range(num_homographyies)]
        self.detector = detector

    def generate_pseudo_label(self, img: np.ndarray) -> np.ndarray:
        pseudo_label = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        for homography in self.homographies:
            warped_img = homography.apply(img)
            point_img = self.detector(warped_img)
            pseudo_label += homography.apply_inverse(point_img)
        return np.heaviside(pseudo_label-0.5, 1)
