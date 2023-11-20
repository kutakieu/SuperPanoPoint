from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


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
