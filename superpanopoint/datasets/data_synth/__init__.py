import json
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from superpanopoint import Settings

from .shapes.base import Shape
from .shapes.utils import generate_background


class SynthData:
    def __init__(self, img_w: int, img_h: int) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.bg_img = generate_background(img_w, img_h)
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
