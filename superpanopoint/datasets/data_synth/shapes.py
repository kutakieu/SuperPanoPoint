import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from superpanopoint import Settings

from .utils import angle_between_vectors, get_random_color

random_state = np.random.RandomState(None)

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
        self.export_points(out_dir / Settings().points_dir_name / f"{sample_id}.txt")
    
    def export_img(self, path: Union[str, Path]):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.synth_img).save(path)

    def export_points(self, path: Union[str, Path]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for shape in self.added_shapes:
                for point in shape.points:
                    f.write(f"{point.x},{point.y}\n")


def generate_background(img_w: int, img_h: int, nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=50, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    dim = max((img_h, img_w))
    cv2.randu(img, 0, 255)
    cv2.threshold(img, random_state.randint(256), 255, cv2.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate([random_state.randint(0, img_w, size=(nb_blobs, 1)),
                            random_state.randint(0, img_h, size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv2.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(int(dim * min_rad_ratio),
                                    int(dim * max_rad_ratio)),
                  col, -1)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv2.blur(img, (kernel_size, kernel_size), img)
    return img


@dataclass
class Point2D:
    x: float
    y: float

    def as_xy(self):
        return [self.x, self.y]


@dataclass
class Segment2D:
    beg: Point2D
    end: Point2D


class Shape(ABC):
    points: Optional[List[Point2D]] = None

    @abstractmethod
    def draw(self, image) -> bool:
        """return True if the shape was drawn, False otherwise"""
        raise NotImplementedError
    
    @abstractmethod
    def drawing_coords(self):
        raise NotImplementedError

    def is_overlap(self, drawing_coords: np.ndarray, cur_img: np.ndarray, bg_img: np.ndarray):
        rows, cols = drawing_coords
        return not np.array_equal(cur_img[rows, cols], bg_img[rows, cols]) 
    
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
        
        col = get_random_color(int(np.mean(img)))
        corners = np.array([p.as_xy() for p in self.points], dtype=int)
        cv2.fillPoly(img, [corners], col)
        return True

    def drawing_coords(self, img: np.ndarray):
        tmp_img = np.zeros(img.shape[:2])
        corners = np.array([p.as_xy() for p in self.points], dtype=int)
        cv2.fillPoly(tmp_img, [corners], 1)
        return np.where(tmp_img > 0)
            

class Star(Shape):
    pass

class Checkerboard(Shape):
    pass

class Stripe(Shape):
    pass

class Cube(Shape):
    pass
