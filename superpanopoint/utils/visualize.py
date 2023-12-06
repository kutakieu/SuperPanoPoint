from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from superpanopoint.datasets import DataSample


def vis_points(img: Union[Image.Image, np.ndarray], points: Union[dict, np.ndarray], color=(0, 255, 255))->np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img.copy()
    
    if isinstance(points, dict):
        points = []
        for item in points["points"]:
            points.append([item["x"], item["y"]])
        points = np.array(points)
    elif img.shape[:2] == points.shape[:2]:
        # when point is a binary mask
        points = points if len(points.shape) == 2 else points[:, :, 0]
        ys, xs = np.where(points > 0)
        points = np.array(list(zip(xs, ys)))

    for x, y in points:
        cv2.circle(img, (x, y), 2, color, -1)

    return Image.fromarray(img)

def vis_matching(img1: Union[Image.Image, np.ndarray], 
                 img2: Union[Image.Image, np.ndarray], 
                 points1: np.ndarray,
                 desc1: np.ndarray,
                 points2: np.ndarray,
                 desc2: np.ndarray,
                 ):
    img1 = np.array(img1).copy()
    img2 = np.array(img2).copy()
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in points1]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in points2]

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    return Image.fromarray(cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
    ))
