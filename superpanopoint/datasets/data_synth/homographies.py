from random import choice, randint, uniform
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


def generate_random_homography(img_w: int, img_h: int) -> "TransformHomography":
    """ Generate a random homography matrix """
    homography_mat = np.eye(3)
    for transform in [Translation, Rotation, Scale, Shear, Perspective]:
        homography_mat = homography_mat @ transform(img_w, img_h).matrix
    return TransformHomography(homography_mat)

def _select_random_homography(img_w: int, img_h: int) -> "TransformHomography":
    return choice([
        Translation(img_w, img_h),
        Rotation(img_w, img_h),
        Scale(img_w, img_h),
        Shear(img_w, img_h),
        Perspective(img_w, img_h),
    ])

def apply_homography(img: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """ Apply homography to an image """
    return cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

class TransformHomography:
    matrix: Optional[np.ndarray] = None
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def apply(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)
        return apply_homography(img, self.matrix)
    
    def apply_inverse(self, img: Union[np.ndarray, Image.Image]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return apply_homography(img, np.linalg.inv(self.matrix))
    
class Translation(TransformHomography):
    tx: float
    ty: float
    def __init__(self, img_w: int, img_h: int, tx: Optional[float]=None, ty: Optional[float]=None) -> None:
        self.tx = randint(0, img_w) if tx is None else tx
        self.ty = randint(0, img_h) if ty is None else ty
        self.matrix = np.array([
            [1, 0, self.tx], 
            [0, 1, self.ty], 
            [0, 0, 1]
        ])

class Rotation(TransformHomography):
    center_x: float
    center_y: float
    angle: float
    def __init__(self, img_w: int, img_h: int, angle: Optional[float]=None, center_x: Optional[int]=None, center_y: Optional[int]=None) -> None:
        self.center_x = randint(0, img_w) if center_x is None else center_x
        self.center_y = randint(0, img_h) if center_y is None else center_y
        self.angle = randint(-30, 30) if angle is None else angle
        self.matrix = np.vstack([
            cv2.getRotationMatrix2D((self.center_x, self.center_y), angle, 1),
            np.array([0, 0, 1])
        ])

class Scale(TransformHomography):
    sx: float
    sy: float
    def __init__(self, img_w: int, img_h: int, scale: Optional[float]=None, center_x: Optional[int]=None, center_y: Optional[int]=None) -> None:
        self.center_x = randint(0, img_w) if center_x is None else center_x
        self.center_y = randint(0, img_h) if center_y is None else center_y
        self.scale = scale if scale is not None else uniform(0.5, 2)
        self.matrix = np.vstack([
            cv2.getRotationMatrix2D((self.center_x, self.center_y), 0, self.scale), 
            np.array([0, 0, 1])
        ])

class Shear(TransformHomography):
    sx: float
    sy: float
    def __init__(self, img_w: int, img_h: int, src_pts: Optional[np.ndarray]=None, dst_pts: Optional[np.ndarray]=None) -> None:
        src_pts = np.array([[uniform(0, img_w/3), uniform(0, img_h/3)], [uniform(0, img_w/3), uniform(img_h/3*2, img_h-1)], [uniform(img_w/3*2, img_w-1), uniform(0, img_h/3)]], dtype=np.float32)
        dst_pts = np.array([[uniform(0, img_w/3), uniform(0, img_h/3)], [uniform(0, img_w/3), uniform(img_h/3*2, img_h-1)], [uniform(img_w/3*2, img_w-1), uniform(0, img_h/3)]], dtype=np.float32)
        cv2.getAffineTransform(src_pts, dst_pts)
        self.matrix = np.vstack([
            cv2.getAffineTransform(src_pts, dst_pts), 
            np.array([0, 0, 1])
        ])

class Perspective(TransformHomography):
    def __init__(self, img_w: int, img_h: int, src_pts: Optional[np.ndarray]=None, dst_pts: Optional[np.ndarray]=None) -> None:
        src_pts = np.array([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]], dtype=np.float32)
        dst_pts = np.array([[uniform(0, img_w/3), uniform(0, img_h/3)], [uniform(0, img_w/3), uniform(img_h/3*2, img_h-1)], [uniform(img_w/3*2, img_w-1), uniform(img_h/3*2, img_h-1)], [uniform(img_w/3*2, img_w-1), uniform(0, img_h/3)]], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
