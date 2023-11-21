import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (get_random_color,
                                                             random_state)


@dataclass
class Star(Shape):
    img_width: int
    img_height: int
    thickness: Optional[int] = None
    color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)
    num_branches: Optional[int] = None

    def __post_init__(self):
        if self.num_branches is None:
            self.num_branches = random_state.randint(3, 6)
        if self.points is None or len(self.points) < 3:
            self.points = self.__create_points()
        if self.thickness is None:
            min_dim = min(self.img_height, self.img_width)
            self.thickness = random_state.randint(min_dim * 0.005, min_dim * 0.01)

    def __create_points(self) -> List[Point2D]:
        min_dim = min(self.img_height, self.img_width)
        rad = max(random_state.rand() * min_dim / 2, min_dim / 5)

        # select the center of the circle
        x = random_state.randint(rad, self.img_width - rad)  
        y = random_state.randint(rad, self.img_height - rad)

        # Sample num_branches points inside the circle
        slices = np.linspace(0, 2 * math.pi, self.num_branches + 1)
        angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                for i in range(self.num_branches)]
        points = [[int(x + max(random_state.rand(), 0.3) * rad * math.cos(a)),
                   int(y + max(random_state.rand(), 0.3) * rad * math.sin(a))]
                   for a in angles]
        points = [[x, y]] + points
        return [Point2D(p[0], p[1]) for p in points]

    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        if self.is_overlap(self.drawing_coords(img), img, bg_img):
            return False
        if self.color is None:
            self.color = get_random_color(int(np.mean(bg_img)))

        for i in range(1, self.num_branches + 1):
            cv2.line(img, 
                     (self.points[0].x, self.points[0].y), (self.points[i].x, self.points[i].y),
                     self.color, self.thickness
                     )
        return True

    def drawing_coords(self, img: np.ndarray):
        tmp_img = np.zeros(img.shape[:2])
        for i in range(1, self.num_branches + 1):
            cv2.line(tmp_img, 
                     (self.points[0].x, self.points[0].y), (self.points[i].x, self.points[i].y),
                     1, self.thickness
                     )
        return np.where(tmp_img > 0)


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background
    
    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Star(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_star.png", img)
