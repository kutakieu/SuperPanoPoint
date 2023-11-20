from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (get_random_color,
                                                             random_state)


@dataclass
class Line(Shape):
    img_width: int
    img_height: int
    thickness: Optional[int] = None
    color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)

    def __post_init__(self):
        if self.color is None:
            self.color = get_random_color(0)
        if self.points is None or len(self.points) < 2:
            self.points = self.__create_points()
        min_dim = min(self.img_height, self.img_width)
        self.thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)

    def __create_points(self) -> (Point2D, Point2D):
        x1 = random_state.randint(self.img_width)
        y1 = random_state.randint(self.img_height)
        p1 = Point2D(x1, y1)
        x2 = random_state.randint(self.img_width)
        y2 = random_state.randint(self.img_height)
        p2 = Point2D(x2, y2)
        return [p1, p2]

    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        if self.is_overlap(self.drawing_coords(img), img, bg_img):
            return False
        cv2.line(img, self.points[0].as_xy(), self.points[1].as_xy(), self.color, self.thickness)
        return True
    
    def drawing_coords(self, img: np.ndarray):
        tmp_img = np.zeros(img.shape[:2])
        cv2.line(tmp_img, self.points[0].as_xy(), self.points[1].as_xy(), 1, self.thickness)
        return np.where(tmp_img > 0)


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background
    
    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Line(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_line.png", img)
