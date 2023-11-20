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
            self.points = self.__create_points()
            
    def __create_points(self) -> List[Point2D]:
        s = min((self.img_width - 1) // self.cols, (self.img_height - 1) // self.rows)  # size of a cell
        num_total_points = (self.rows + 1) * (self.cols + 1)
        x_coord = np.tile(range(self.cols + 1), self.rows + 1).reshape((num_total_points, 1))
        y_coord = np.repeat(range(self.rows + 1), self.cols + 1).reshape((num_total_points, 1))
        xys = s * np.concatenate([x_coord, y_coord], axis=1)

        transform = generate_random_homography(self.img_width, self.img_height)
        return transform.apply_to_coordinates(xys).astype(int)
            
    def ____create_points(self) -> List[Point2D]:
        s = min((self.img_width - 1) // self.cols, (self.img_height - 1) // self.rows)  # size of a cell
        x_coord = np.tile(range(self.cols + 1),
                        self.rows + 1).reshape(((self.rows + 1) * (self.cols + 1), 1))
        y_coord = np.repeat(range(self.rows + 1),
                            self.cols + 1).reshape(((self.rows + 1) * (self.cols + 1), 1))
        self.points = points = s * np.concatenate([x_coord, y_coord], axis=1)
        # self.warped_points = self.points

        # Warp the grid using an affine transformation and an homography
        # The parameters of the transformations are constrained
        # to get transformations not too far-fetched
        transform_params=(0.05, 0.15)
        alpha_affine = np.max(img.shape) * (transform_params[0]
                                            + random_state.rand() * transform_params[1])
        center_square = np.float32(img.shape) // 2
        min_dim = min(img.shape)
        square_size = min_dim // 3
        pts1 = np.float32([center_square + square_size,
                        [center_square[0]+square_size, center_square[1]-square_size],
                        center_square - square_size,
                        [center_square[0]-square_size, center_square[1]+square_size]])
        pts2 = pts1 + random_state.uniform(-alpha_affine,
                                        alpha_affine,
                                        size=pts1.shape).astype(np.float32)
        affine_transform = cv2.getAffineTransform(pts1[:3], pts2[:3])
        pts2 = pts1 + random_state.uniform(-alpha_affine / 2,
                                        alpha_affine / 2,
                                        size=pts1.shape).astype(np.float32)
        perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the affine transformation
        points = np.transpose(np.concatenate((points,
                                            np.ones(((self.rows + 1) * (self.cols + 1), 1))),
                                            axis=1))
        warped_points = np.transpose(np.dot(affine_transform, points))

        # Apply the homography
        warped_col0 = np.add(np.sum(np.multiply(warped_points,
                                                perspective_transform[0, :2]), axis=1),
                            perspective_transform[0, 2])
        warped_col1 = np.add(np.sum(np.multiply(warped_points,
                                                perspective_transform[1, :2]), axis=1),
                            perspective_transform[1, 2])
        warped_col2 = np.add(np.sum(np.multiply(warped_points,
                                                perspective_transform[2, :2]), axis=1),
                            perspective_transform[2, 2])
        warped_col0 = np.divide(warped_col0, warped_col2)
        warped_col1 = np.divide(warped_col1, warped_col2)
        warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
        self.warped_points = warped_points.astype(int)

    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        print(self.points.shape)
        background_color = int(np.mean(bg_img))
        min_dim = min(self.img_height, self.img_width)

        # Fill the rectangles
        colors = np.zeros((self.rows * self.cols,), np.int32)
        for i in range(self.rows):
            for j in range(self.cols):
                # Get a color that contrast with the neighboring cells
                if i == 0 and j == 0:
                    col = get_random_color(background_color)
                else:
                    neighboring_colors = []
                    if i != 0:
                        neighboring_colors.append(colors[(i-1) * self.cols + j])
                    if j != 0:
                        neighboring_colors.append(colors[i * self.cols + j - 1])
                    col = _get_different_color(np.array(neighboring_colors))
                colors[i * self.cols + j] = col
                # Fill the cell
                cv2.fillConvexPoly(img, np.array([(self.points[i * (self.cols + 1) + j, 0],
                                                self.points[i * (self.cols + 1) + j, 1]),
                                                (self.points[i * (self.cols + 1) + j + 1, 0],
                                                self.points[i * (self.cols + 1) + j + 1, 1]),
                                                (self.points[(i + 1)
                                                                * (self.cols + 1) + j + 1, 0],
                                                self.points[(i + 1)
                                                                * (self.cols + 1) + j + 1, 1]),
                                                (self.points[(i + 1)
                                                                * (self.cols + 1) + j, 0],
                                                self.points[(i + 1)
                                                                * (self.cols + 1) + j, 1])]),
                                col)

        # Draw lines on the boundaries of the board at random
        nb_rows = random_state.randint(2, self.rows + 2)
        nb_cols = random_state.randint(2, self.cols + 2)
        thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
        for _ in range(nb_rows):
            row_idx = random_state.randint(self.rows + 1)
            col_idx1 = random_state.randint(self.cols + 1)
            col_idx2 = random_state.randint(self.cols + 1)
            col = get_random_color(background_color)
            cv2.line(img, (self.points[row_idx * (self.cols + 1) + col_idx1, 0],
                        self.points[row_idx * (self.cols + 1) + col_idx1, 1]),
                    (self.points[row_idx * (self.cols + 1) + col_idx2, 0],
                    self.points[row_idx * (self.cols + 1) + col_idx2, 1]),
                    col, thickness)
        for _ in range(nb_cols):
            col_idx = random_state.randint(self.cols + 1)
            row_idx1 = random_state.randint(self.rows + 1)
            row_idx2 = random_state.randint(self.rows + 1)
            col = get_random_color(background_color)
            cv2.line(img, (self.points[row_idx1 * (self.cols + 1) + col_idx, 0],
                        self.points[row_idx1 * (self.cols + 1) + col_idx, 1]),
                    (self.points[row_idx2 * (self.cols + 1) + col_idx, 0],
                    self.points[row_idx2 * (self.cols + 1) + col_idx, 1]),
                    col, thickness)

        # Keep only the points inside the image
        points = keep_points_inside(self.points, img.shape[:2])
        return points
    
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
