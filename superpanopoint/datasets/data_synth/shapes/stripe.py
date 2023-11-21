from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.homographies import \
    generate_random_homography
from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (
    get_random_color, keep_points_inside, random_fillPoly, random_state)


@dataclass
class Stripe(Shape):
    img_width: int
    img_height: int
    background_color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)
    points_for_drawing: np.ndarray = field(init=False)  # shape: (num_points, 2)
    max_nb_cols: int=13
    min_width_ratio: float=0.04


    def __post_init__(self):
        if self.background_color is None:
            self.color = get_random_color(0)
        if not self.points:
            self.points_for_drawing, self.points = self.__create_points()


    def __create_points(self) -> List[Point2D]:
        # Create the grid
        board_w, board_h = int(self.img_width * (1 + random_state.rand())), int(self.img_height * (1 + random_state.rand()))
        self.n_cols = random_state.randint(5, self.max_nb_cols)  # number of cols
        col_idxs = np.concatenate([board_w * random_state.rand(self.n_cols - 1), np.array([0, board_w - 1])], axis=0)
        col_idxs = np.unique(col_idxs.astype(int))  # cols are sorted
        
        # Remove the indices that are too close
        min_dim = min(self.img_height, self.img_width)
        min_width = min_dim * self.min_width_ratio
        col_idxs = col_idxs[(np.concatenate([col_idxs[1:], np.array([board_w + min_width])], axis=0) - col_idxs) >= min_width]
        self.n_cols = col_idxs.shape[0] - 1  # update the number of cols

        xs = np.reshape(col_idxs, (self.n_cols + 1, 1))
        xys_top = np.concatenate([xs, np.zeros((self.n_cols+1, 1), int)], axis=1)
        xys_bottom = np.concatenate([xs, np.full((self.n_cols+1, 1), board_h-1, int)], axis=1)
        xys = np.concatenate([xys_top, xys_bottom], axis=0)
        
        transform = generate_random_homography(self.img_width, self.img_height)
        points = transform.apply_to_coordinates(xys).astype(int)
        return points, [Point2D(p[0], p[1]) for p in keep_points_inside(points, self.img_width, self.img_height)]


    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        background_color = int(np.mean(bg_img))
        min_dim = min(self.img_height, self.img_width)

        # Fill the rectangles
        color = get_random_color(background_color)
        for i in range(self.n_cols):
            color = (color + 128 + random_state.randint(-30, 30)) % 256
            random_fillPoly(img,
                            np.array([
                                (self.points_for_drawing[i, 0], self.points_for_drawing[i, 1]),
                                (self.points_for_drawing[i+1, 0], self.points_for_drawing[i+1, 1]),
                                (self.points_for_drawing[i+self.n_cols+2, 0], self.points_for_drawing[i+self.n_cols+2, 1]),
                                (self.points_for_drawing[i+self.n_cols+1, 0], self.points_for_drawing[i+self.n_cols+1, 1])
                                ]),
                            color)

        # Draw lines on the boundaries of the stripes at random
        nb_rows = random_state.randint(2, 5)
        nb_cols = random_state.randint(2, self.n_cols + 2)
        thickness = random_state.randint(min_dim * 0.005, min_dim * 0.01)
        for _ in range(nb_rows):
            row_idx = random_state.choice([0, self.n_cols + 1])
            col_idx1 = random_state.randint(self.n_cols + 1)
            col_idx2 = random_state.randint(self.n_cols + 1)
            color = get_random_color(background_color)
            cv2.line(img, 
                     (self.points_for_drawing[row_idx + col_idx1, 0], self.points_for_drawing[row_idx + col_idx1, 1]),
                     (self.points_for_drawing[row_idx + col_idx2, 0], self.points_for_drawing[row_idx + col_idx2, 1]),
                     color, thickness)
        for _ in range(nb_cols):
            col_idx = random_state.randint(self.n_cols + 1)
            color = get_random_color(background_color)
            cv2.line(img, 
                     (self.points_for_drawing[col_idx, 0], self.points_for_drawing[col_idx, 1]),
                     (self.points_for_drawing[col_idx + self.n_cols + 1, 0], self.points_for_drawing[col_idx + self.n_cols + 1, 1]),
                     color, thickness)
        return True


    def drawing_coords(self):
        raise NotImplementedError


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background

    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Stripe(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_stripe.png", img)
    # from superpanopoint.datasets.synthetic.synthetic_dataset import \
    #     draw_stripes
    # img = generate_background(IMG_W, IMG_H)
    # draw_stripes(img)
    # cv2.imwrite("img_stripe.png", img)
