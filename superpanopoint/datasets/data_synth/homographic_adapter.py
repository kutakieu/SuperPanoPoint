from random import randint
from typing import List

import cv2
import numpy as np
from PIL import Image

from superpanopoint import Settings
from superpanopoint.datasets.pano.e2p import e2p_map, e2p_with_map
from superpanopoint.predictors import MagicPointPredictor
from superpanopoint.predictors.postprocess import non_maximum_suppression

from .homographies import Perspective, Rotation, Scale, TransformHomography


class HomographicAdapter:
    def __init__(self, img_w: int, img_h: int, detector: MagicPointPredictor, num_homographies: int=100) -> None:
        self.homographies = self._make_basic_homographies(img_w, img_h)
        self.homographies += [self._generate_random_homography(img_w, img_h) for _ in range(num_homographies-len(self.homographies))]
        self.detector = detector
    
    def _make_basic_homographies(self, img_w: int, img_h: int, n_cells:int = 3) -> List[TransformHomography]:
        homographies = [
            TransformHomography(np.eye(3), img_w, img_h)
        ]
        cell_w, cell_h = img_w//n_cells, img_h//n_cells
        cx, cy = img_w//(n_cells*2), img_h//(n_cells*2)
        homographies += [Scale(img_w, img_h, scale=2, center_x=cx+c*cell_w, center_y=cy+r*cell_h) for r in range(3) for c in range(3)]
        return homographies

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


class PanoramaTransformAdapter:
    def __init__(self, detector: MagicPointPredictor, fov_deg: int=45, pano_hw=[1024, 2048], pers_hw=[256, 256]) -> None:
        self.detector = detector
        self.fov_deg = fov_deg
        self.pano_hw = pano_hw
        self.pers_hw = pers_hw
        self.e2p_maps = self._calc_e2p_maps()

    def _calc_e2p_maps(self):
        e2p_maps = {}
        for u_deg in np.linspace(-180, 180, 18, endpoint=False):
            for v_deg in np.linspace(-45, 45, 5):
                e2p_maps[(u_deg, v_deg)] = e2p_map(in_hw=self.pano_hw, fov_deg=self.fov_deg, u_deg=u_deg, v_deg=v_deg, out_hw=self.pers_hw)
        return e2p_maps
    
    def generate_pseudo_labels(self, pano_img: np.ndarray):
        pano_img = np.array(Image.fromarray(pano_img).resize(self.pano_hw[::-1]))

        points_on_pano = []
        for u_deg in np.linspace(-180, 180, 18, endpoint=False):
            for v_deg in np.linspace(-45, 45, 5):
                pers_img = e2p_with_map(pano_img, self.e2p_maps[(u_deg, v_deg)])
                points = self.detector(pers_img, return_as_array=True)
                for x, y in points:
                    points_on_pano.append(np.array(self.e2p_maps[(u_deg, v_deg)][y, x]).round().astype(int))

        return {
            Settings().img_width_key: self.pano_hw[1],
            Settings().img_height_key: self.pano_hw[0],
            Settings().points_key: [{"x": x, "y": y} for x, y in points_on_pano]
        }
