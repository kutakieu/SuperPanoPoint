import json
from pathlib import Path
from random import choice, randint
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from superpanopoint import Settings
from superpanopoint.datasets.pano.c2e import c2e
from superpanopoint.datasets.pano.utils import uv2coor, xyz2uv, xyzcube

from .shapes import Checkerboard, Cube, Line, Polygon, Star, Stripe
from .shapes.base import Shape
from .shapes.utils import generate_background

NUM_MIN_LINES = 3
NUM_MAX_LINES = 10
NUM_MIN_POLYGONS = 10
NUM_MAX_POLYGONS = 10
NUM_MIN_STARS = 1
NUM_MAX_STARS = 3


def generate_perspective_sample(img_w: int, img_h: int) -> "SynthData":
    synth_data = SynthData(img_w, img_h)
    if randint(0, 1) == 0:
        adding_shape = choice([Cube, Checkerboard, Stripe])
        synth_data.add_shapes(adding_shape, 1)

    synth_data.add_shapes(Line, randint(NUM_MIN_LINES, NUM_MAX_LINES))
    synth_data.add_shapes(Polygon, randint(NUM_MIN_POLYGONS, NUM_MAX_POLYGONS))
    synth_data.add_shapes(Star, randint(NUM_MIN_STARS, NUM_MAX_STARS))
    return synth_data


class SynthData:
    def __init__(self, img_w: int, img_h: int, bg_img: Optional[np.ndarray]=None) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.bg_img = generate_background(img_w, img_h) if bg_img is None else bg_img
        self.synth_img = self.bg_img.copy()
        self.added_shapes: List[Shape] = []

    def add_shapes(self, shape_type: "Shape", num_shpaes: int):
        for _ in range(num_shpaes):
            shape = shape_type(img_width=self.img_w, img_height=self.img_h)
            self.draw_shape(shape)

    def draw_shape(self, shape: "Shape"):
        if shape.draw(self.synth_img, self.bg_img):
            self.added_shapes.append(shape)        

    def export(self, out_dir: Union[str, Path], sample_id: str):
        self.export_img(out_dir / Settings().img_dir_name / f"{sample_id}.png")
        self.export_points(out_dir / Settings().points_dir_name / f"{sample_id}.json")
    
    def export_img(self, path: Union[str, Path], with_points: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if with_points:
            new_img = self.synth_img.copy()
            new_img = np.repeat(new_img[:, :, np.newaxis], 3, axis=2)
            for shape in self.added_shapes:
                for point in shape.points:
                    cv2.circle(new_img, (int(point.x), int(point.y)), 2, (255, 0, 0), 1)
            Image.fromarray(new_img).save(path)
        else:
            Image.fromarray(self.synth_img).save(path)

    def export_points(self, path: Union[str, Path]):
        path.parent.mkdir(parents=True, exist_ok=True)
        points_dict = {
            Settings().img_width_key: self.img_w,
            Settings().img_height_key: self.img_h,
            Settings().points_key: [{"x": int(point.x), "y": int(point.y)} for shape in self.added_shapes for point in shape.points]
        }
        with open(path, "w") as f:
            json.dump(points_dict, f, indent=4)

    def points_as_img(self) -> np.ndarray:
        point_img = np.zeros((self.img_h, self.img_w, 1), dtype=float)
        for shape in self.added_shapes:
            for point in shape.points:
                point_img[point.y, point.x, :] = 1
        return point_img


class PanoSynthData:
    def __init__(self, synth_data_list: List[SynthData], pers_size: int=512, pano_w: int=2048, pano_h: int=1024) -> None:
        self.synth_data_list = synth_data_list
        self.pers_size = pers_size
        self.pano_w, self.pano_h = pano_w, pano_h
        self.synth_pano_img = self._pers_imgs_to_pano()
        self.points = self.points_on_pano(synth_data_list)

    def _pers_imgs_to_pano(self) -> np.ndarray:
        pano_data = np.zeros((self.pers_size, self.pers_size*6), dtype=np.uint8)
        for i, synth_data in enumerate(self.synth_data_list):
            pano_data[:, i*self.pers_size:(i+1)*self.pers_size] = synth_data.synth_img
        return c2e(pano_data, self.pano_h, self.pano_w, mode='bilinear', cube_format='horizon').astype(np.uint8)[:, :, 0]
    
    def points_on_pano(self, synth_data_list: List[SynthData]):
        xyz = xyzcube(self.pers_size)
        uv = xyz2uv(xyz)
        coor_xy = uv2coor(uv, self.pano_h, self.pano_w)
        points = []
        for i, synth_data in enumerate(synth_data_list):
            start_col = i * self.pers_size
            for shape in synth_data.added_shapes:
                for point in shape.points:
                    col, row = start_col + point.x, point.y
                    points.append(coor_xy[row, col])
        return np.array(points)

    def export(self, out_dir: Union[str, Path], sample_id: str):
        self.export_img(out_dir / Settings().img_dir_name / f"{sample_id}.png")
        self.export_points(out_dir / Settings().points_dir_name / f"{sample_id}.json")

    def export_img(self, path: Union[str, Path], with_points: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if with_points:
            new_img = self.synth_pano_img.copy()
            new_img = np.repeat(new_img[:, :, np.newaxis], 3, axis=2)
            for point in self.points:
                cv2.circle(new_img, (int(point[0]), int(point[1])), 2, (255, 0, 0), 1)
            Image.fromarray(new_img).save(path)
        else:
            Image.fromarray(self.synth_pano_img).save(path)

    def export_points(self, path: Union[str, Path]):
        path.parent.mkdir(parents=True, exist_ok=True)
        points_dict = {
            Settings().img_width_key: self.pano_w,
            Settings().img_height_key: self.pano_h,
            Settings().points_key: [{"x": int(point[0]), "y": int(point[1])} for point in self.points]
        }
        with open(path, "w") as f:
            json.dump(points_dict, f, indent=4)

