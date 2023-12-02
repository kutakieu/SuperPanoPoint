import math
from dataclasses import dataclass, field
from random import randint
from typing import List, Optional, Tuple

import cv2
import numpy as np

from superpanopoint.datasets.data_synth.shapes.base import Point2D, Shape
from superpanopoint.datasets.data_synth.shapes.utils import (
    get_random_color, keep_points_inside, random_fillPoly, random_state)


@dataclass
class Cube(Shape):
    img_width: int
    img_height: int
    background_color: Optional[Tuple[int]] = None
    points: List[Point2D] = field(default_factory=list)
    points_for_drawing: np.ndarray = field(init=False)  # shape: (num_points, 2)
    min_size_ratio: float=0.2
    scale_interval=(0.3, 0.4)
    trans_interval=(0.3, 0.1)
    only_perfect: bool=True  # only cubes with all the corners visible


    def __post_init__(self):
        self.points_for_drawing, self.points = self.__create_points()


    def __create_points(self) -> List[Point2D]:
        cube_xys = self._generate_cube_vertices().astype(int)
        points = keep_points_inside(cube_xys[1:, :], self.img_height, self.img_width) # get rid of the hidden corner
        return cube_xys, [Point2D(p[0], p[1]) for p in points]


    def draw(self, img: np.ndarray, bg_img: np.ndarray):
        if self.only_perfect and len(self.points) < 7:  # not enough corners
            return False
        # Get the three visible faces
        faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])
        min_dim = min(self.img_height, self.img_width)
        # Fill the faces and draw the contours
        face_col = get_random_color(int(np.mean(bg_img)))
        for i in [0, 1, 2]:
            cur_face_col = max(0, min(255, face_col + random_state.randint(-15, 15)))
            random_fillPoly(img, self.points_for_drawing[faces[i]].reshape((-1, 1, 2)), cur_face_col)
        thickness = max(1, random_state.randint(min_dim * 0.003, min_dim * 0.01))
        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3]:
                col_edge = (face_col + 128 + random_state.randint(-64, 64)) % 256  # color that constrats with the face color
                cv2.line(img, 
                         (self.points_for_drawing[faces[i][j], 0], self.points_for_drawing[faces[i][j], 1]),
                         (self.points_for_drawing[faces[i][(j + 1) % 4], 0], self.points_for_drawing[faces[i][(j + 1) % 4], 1]),
                         col_edge, thickness)
        return True


    def drawing_coords(self):
        raise NotImplementedError


    def _generate_cube_vertices(self) -> np.ndarray:
        min_dim = min(self.img_height, self.img_width)
        min_side = min_dim * self.min_size_ratio
        lx = min_side + random_state.rand() * 2 * min_dim / 3  # dimensions of the cube
        ly = min_side + random_state.rand() * 2 * min_dim / 3
        lz = min_side + random_state.rand() * 2 * min_dim / 3

        cube_xyzs = np.array([
                            [0, 0, 0],
                            [lx, 0, 0],
                            [0, ly, 0],
                            [lx, ly, 0],
                            [0, 0, lz],
                            [lx, 0, lz],
                            [0, ly, lz],
                            [lx, ly, lz]
                            ])

        rot_angles = random_state.rand(3) * 3 * math.pi / 10. + math.pi / 10.
        rotation_1 = np.array([
                            [math.cos(rot_angles[0]), -math.sin(rot_angles[0]), 0],
                            [math.sin(rot_angles[0]), math.cos(rot_angles[0]), 0],
                            [0, 0, 1]
                            ])
        rotation_2 = np.array([
                            [1, 0, 0],
                            [0, math.cos(rot_angles[1]), -math.sin(rot_angles[1])],
                            [0, math.sin(rot_angles[1]), math.cos(rot_angles[1])]
                            ])
        rotation_3 = np.array([
                            [math.cos(rot_angles[2]), 0, -math.sin(rot_angles[2])],
                            [0, 1, 0],
                            [math.sin(rot_angles[2]), 0, math.cos(rot_angles[2])]
                            ])
        scaling = np.array([
                            [self.scale_interval[0] + random_state.rand() * self.scale_interval[1], 0, 0],
                            [0, self.scale_interval[0] + random_state.rand() * self.scale_interval[1], 0],
                            [0, 0, self.scale_interval[0] + random_state.rand() * self.scale_interval[1]]
                            ])
        trans_x = self.img_width * self.trans_interval[0] + \
            random_state.randint(-self.img_width * self.trans_interval[1], self.img_width * self.trans_interval[1])
        trans_y = self.img_height * self.trans_interval[0] + \
            random_state.randint(-self.img_height * self.trans_interval[1], self.img_height * self.trans_interval[1])
        trans = np.array([trans_x, trans_y, 0])
        cube_xyzs = trans + np.transpose(np.dot(scaling,
                                        np.dot(rotation_1,
                                                np.dot(rotation_2,
                                                        np.dot(rotation_3,
                                                                np.transpose(cube_xyzs))))))
        # The hidden corner is 0 by construction
        # The front one is 7
        cube_xys = cube_xyzs[:, :2]  # project on the plane z=0
        return cube_xys.astype(int)


if __name__ == "__main__":
    from superpanopoint.datasets.data_synth.shapes.utils import \
        generate_background

    IMG_W, IMG_H = 512, 512
    bg_img = generate_background(IMG_W, IMG_H)
    img = bg_img.copy()
    shape = Cube(IMG_W, IMG_H)
    shape.draw(img, bg_img)
    cv2.imwrite("img_cube.png", img)
    # from superpanopoint.datasets.synthetic.synthetic_dataset import draw_cube
    # img = generate_background(IMG_W, IMG_H)
    # draw_cube(img)
    # cv2.imwrite("img_cube.png", img)
