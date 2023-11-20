from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.homographies import \
    generate_random_homography
from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (
    get_random_color, keep_points_inside, random_state)


@dataclass
class Checkerboard(Shape):
    img_width: int
    img_height: int
    background_color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)
    points_for_drawing: np.ndarray = field(init=False)  # shape: (num_points, 2)
    rows: Optional[int] = None
    cols: Optional[int] = None


    def __post_init__(self):
        if self.background_color is None:
            self.color = get_random_color(0)
        if self.rows is None:
            self.rows = random_state.randint(3, 7)
        if self.cols is None:
            self.cols = random_state.randint(3, 7)
        if not self.points:
            self.points_for_drawing, self.points = self.__create_points()


    def __create_points(self) -> List[Point2D]:
        cell_size = min((self.img_width - 1) // self.cols, (self.img_height - 1) // self.rows)  # size of a cell
        num_total_points = (self.rows + 1) * (self.cols + 1)
        x_coord = np.tile(range(self.cols + 1), self.rows + 1).reshape((num_total_points, 1))
        y_coord = np.repeat(range(self.rows + 1), self.cols + 1).reshape((num_total_points, 1))
        xys = cell_size * np.concatenate([x_coord, y_coord], axis=1)

        transform = generate_random_homography(self.img_width, self.img_height)
        points = transform.apply_to_coordinates(xys).astype(int)
        return points, [Point2D(p[0], p[1]) for p in keep_points_inside(points, self.img_width, self.img_height)]


    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        background_color = int(np.mean(bg_img))
        points_2d = self.points_for_drawing.reshape(self.rows + 1, self.cols + 1, 2)

        # Fill the rectangles
        colors = np.zeros((self.rows * self.cols,), np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                # Get a color that contrast with the neighboring cells
                if r == 0 and c == 0:
                    col = get_random_color(background_color)
                else:
                    neighboring_colors = []
                    if r != 0:
                        neighboring_colors.append(colors[(r-1) * self.cols + c])
                    if c != 0:
                        neighboring_colors.append(colors[r * self.cols + c - 1])
                    col = _get_different_color(np.array(neighboring_colors))
                colors[r * self.cols + c] = col
                # Fill the cell
                cv2.fillConvexPoly(img,
                                   np.array([
                                       (points_2d[r, c, 0], points_2d[r, c, 1]),
                                       (points_2d[r, c+1, 0], points_2d[r, c+1, 1]),
                                       (points_2d[r+1, c+1, 0], points_2d[r+1, c+1, 1]),
                                       (points_2d[r+1, c, 0], points_2d[r+1, c, 1])]),
                                    col)

        # Draw lines on the boundaries of the board at random
        nb_rows = random_state.randint(2, self.rows + 2)
        nb_cols = random_state.randint(2, self.cols + 2)
        min_dim = min(self.img_height, self.img_width)
        thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
        for _ in range(nb_rows):
            r = random_state.randint(self.rows + 1)
            c1 = random_state.randint(self.cols + 1)
            c2 = random_state.randint(self.cols + 1)
            col = get_random_color(background_color)
            cv2.line(img, 
                    (points_2d[r, c1, 0], points_2d[r, c1, 1]),
                    (points_2d[r, c2, 0], points_2d[r, c2, 1]),
                    col, thickness)
        for _ in range(nb_cols):
            c = random_state.randint(self.cols + 1)
            r1 = random_state.randint(self.rows + 1)
            r2 = random_state.randint(self.rows + 1)
            col = get_random_color(background_color)
            cv2.line(img, 
                    (points_2d[r1, c, 0], points_2d[r1, c, 1]),
                    (points_2d[r2, c, 0], points_2d[r2, c, 1]),
                    col, thickness)
        return True


    def drawing_coords(self):
        raise NotImplementedError


def _get_different_color(previous_colors, min_dist=50, max_count=20):
    """ Output a color that contrasts with the previous colors
    Parameters:
      previous_colors: np.array of the previous colors
      min_dist: the difference between the new color and
                the previous colors must be at least min_dist
      max_count: maximal number of iterations
    """
    color = random_state.randint(256)
    count = 0
    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = random_state.randint(256)
    return color


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background
    
    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Checkerboard(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_checkerboard.png", img)
