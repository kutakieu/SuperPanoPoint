from random import randint
from typing import List

import cv2
import numpy as np
from PIL import Image

from superpanopoint import Settings
from superpanopoint.models.predictor import MagicPointPredictor
from superpanopoint.models.utils.postprocess import non_maximum_suppression

from .homographies import Perspective, Rotation, Scale, TransformHomography


class HomographicAdapter:
    def __init__(self, img_w: int, img_h: int, detector: MagicPointPredictor, num_homographies: int=100) -> None:
        self.homographies: List[TransformHomography] = [self._generate_random_homography(img_w, img_h) for _ in range(num_homographies)]
        self.detector = detector

    def _generate_random_homography(self, img_w: int, img_h: int) -> TransformHomography:
        homography_mat = np.eye(3)
        for transform in [Scale(img_w, img_h, scale=randint(4,6)), Rotation(img_w, img_h), Perspective(img_w, img_h)]:
            homography_mat = homography_mat @ transform.matrix
        return TransformHomography(homography_mat, img_w, img_h)

    def generate_pseudo_labels(self, img: np.ndarray, prob_thres: float=1.0, omit_edge_width: int=8, asjson: bool=True) -> np.ndarray:
        orig_h, orig_w = img.shape[:2]
        total_prob_map = np.zeros((orig_h, orig_w), dtype=float)
        for homography in self.homographies:
            warped_img = homography.apply(img)
            warped_img = np.array(Image.fromarray(warped_img.astype(np.uint8)).resize((256, 256)))
            prob_map = self.detector.calc_prob_map(warped_img, omit_edge_width=omit_edge_width//2)
            prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            total_prob_map += homography.apply_inverse(prob_map)

        total_prob_map[total_prob_map < prob_thres] = 0
        total_prob_map[:, :omit_edge_width] = 0
        total_prob_map[:, -omit_edge_width:] = 0
        total_prob_map[:omit_edge_width, :] = 0
        total_prob_map[-omit_edge_width:, :] = 0

        pseudo_label = non_maximum_suppression(total_prob_map)

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
