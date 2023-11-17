from random import choice, randint, uniform
from typing import Optional, Union

import cv2
import numpy as np
from einops import rearrange
from PIL import Image


def generate_random_homography(img_w: int, img_h: int) -> "TransformHomography":
    """ Generate a random homography matrix """
    homography_mat = np.eye(3)
    for transform in [Translation, Rotation, Scale, Perspective]:
        homography_mat = homography_mat @ transform(img_w, img_h).matrix
    return TransformHomography(homography_mat, img_w, img_h)

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
    def __init__(self, matrix: np.ndarray, img_w: int, img_h: int) -> None:
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)
        self.img_w = img_w
        self.img_h = img_h

    def apply(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)
        return apply_homography(img, self.matrix)
    
    def apply_inverse(self, img: Union[np.ndarray, Image.Image]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return apply_homography(img, self.inverse_matrix)
    
    def make_mask(self, inverse=False) -> np.ndarray:
        xs, ys = np.meshgrid(np.arange(512), np.arange(512))
        homogeneous_coords = np.concatenate([xs[:,:,np.newaxis], ys[:,:,np.newaxis], np.ones((self.img_w, self.img_h, 1))], axis=2)
        mat = self.inverse_matrix if inverse else self.matrix
        converted_coords = (mat @ homogeneous_coords.reshape(-1, 3).T).T

        # normalizee coordinates by dividing by the last element of homogeneous coordinates(w)
        w = 1 / converted_coords[:, 2]
        w = np.repeat(w[:, np.newaxis], 3, axis=1)
        normalized_coords = np.multiply(converted_coords, w.reshape(-1, 3))

        return (
            (0 <= normalized_coords[:, 0]) & 
            (normalized_coords[:, 0] < self.img_h) & 
            (0 <= normalized_coords[:, 1]) & 
            (normalized_coords[:, 1] < self.img_w)
            ).reshape(self.img_h, self.img_w).astype(np.uint8)
    
    def get_transformed_indices(self, inverse=False) -> np.ndarray:
        xs, ys = np.meshgrid(np.arange(self.img_w), np.arange(self.img_h))
        homogeneous_coords = np.concatenate([xs[:,:,np.newaxis], ys[:,:,np.newaxis], np.ones((self.img_h, self.img_w, 1))], axis=2)
        mat = self.inverse_matrix if inverse else self.matrix
        converted_coords = (mat @ homogeneous_coords.reshape(-1, 3).T).T

        # normalizee coordinates by dividing by the last element of homogeneous coordinates(w)
        w = 1 / converted_coords[:, 2]
        w = np.repeat(w[:, np.newaxis], 3, axis=1)
        normalized_coords = np.multiply(converted_coords, w.reshape(-1, 3))

        return normalized_coords[:, :2].reshape(self.img_h, self.img_w, 2).astype(np.float32)
    
    def get_correspondence_mask(self, inverse=False, scale: int=8) -> np.ndarray:
        xs, ys = np.meshgrid(np.arange(self.img_w), np.arange(self.img_h))
        original_indices = np.concatenate([xs[:,:,np.newaxis], ys[:,:,np.newaxis]], axis=2)
        transformed_indices = self.get_transformed_indices(inverse)
        diff = np.subtract(
            rearrange(original_indices[::scale, ::scale], "w h d -> (w h) 1 d"), 
            rearrange(transformed_indices[::scale, ::scale], "w h d -> 1 (w h) d")
            )
        return ((np.sum(diff**2, axis=2) ** 0.5) <= (scale//2)).astype(float)
    
class Translation(TransformHomography):
    tx: float
    ty: float
    def __init__(self, img_w: int, img_h: int, tx: Optional[float]=None, ty: Optional[float]=None) -> None:
        self.tx = randint(-img_w//3, img_w//3) if tx is None else tx
        self.ty = randint(-img_h//3, img_h//3) if ty is None else ty
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
        self.center_x = img_w//2 if center_x is None else center_x
        self.center_y = img_h//2 if center_y is None else center_y
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
