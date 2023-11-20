import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (
    angle_between_vectors, get_random_color, random_state)


@dataclass
class Polygon(Shape):
    img_width: int
    img_height: int
    max_sides: int = 8
    color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)

    def __post_init__(self):
        if self.color is None:
            self.color = get_random_color(0)
        if self.points is None or len(self.points) < 3:
            self.points = self.__create_points()

    def __create_points(self) -> List[Point2D]:
        num_corners = random_state.randint(3, self.max_sides)
        min_dim = min(self.img_height, self.img_width)
        rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
        circle_center_x = random_state.randint(rad, self.img_width - rad)
        circle_center_y = random_state.randint(rad, self.img_height - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                for i in range(num_corners)]
        points = np.array([[int(circle_center_x + max(random_state.rand(), 0.4) * rad * math.cos(a)),
                            int(circle_center_y + max(random_state.rand(), 0.4) * rad * math.sin(a))]
                        for a in angles])

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(points[(i-1) % num_corners, :]
                                - points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        points = points[mask, :]
        num_corners = points.shape[0]
        corner_angles = [angle_between_vectors(points[(i-1) % num_corners, :] -
                                            points[i, :],
                                            points[(i+1) % num_corners, :] -
                                            points[i, :])
                        for i in range(num_corners)]
        mask = np.array(corner_angles) < (2 * math.pi / 3)
        points = points[mask, :]
        return [Point2D(p[0], p[1]) for p in points]

    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        if len(self.points) < 3:  # not enough corners
            return False
        if self.is_overlap(self.drawing_coords(img), img, bg_img):
            return False
        
        corners = np.array([p.as_xy() for p in self.points], dtype=int)
        cv2.fillPoly(img, [corners], self.color)
        return True

    def drawing_coords(self, img: np.ndarray):
        tmp_img = np.zeros(img.shape[:2])
        corners = np.array([p.as_xy() for p in self.points], dtype=int)
        cv2.fillPoly(tmp_img, [corners], 1)
        return np.where(tmp_img > 0)


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background
    
    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Polygon(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_polygon.png", img)
