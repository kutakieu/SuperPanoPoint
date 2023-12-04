import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from superpanopoint.datasets.data_synth.homographic_adapter import \
    HomographicAdapter
from superpanopoint.models.predictor import MagicPointPredictor

config_path = "config/config_magicpoint.yaml"
model_path = 'model_magicpoint.pth'

cfg = OmegaConf.load(config_path)
detector = MagicPointPredictor(cfg, model_path, device='cuda')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', help="directory of input images", type=str, default="data/homographic/coco/imgs")
    parser.add_argument('--dir_out', help="directory to save points", type=str, default="data/homographic/coco/points")
    return parser.parse_args()


def main():
    args = get_args()
    dir_in = Path(args.dir_in)
    dir_out = Path(args.dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)
    dir_vis_out = dir_out.parent / 'with_points'
    dir_vis_out.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(dir_in.glob('*'))):
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]
        homographic_adapter = HomographicAdapter(w, h, detector, num_homographies=100)
        points_json = homographic_adapter.generate_pseudo_labels(img, asjson=True)

        vis_img = visualize_points(img, points_json['points'])
        Image.fromarray(vis_img).save(dir_vis_out / f'{img_path.stem}.png')

        with open(dir_out / f'{img_path.stem}.json', 'w') as f:
            json.dump(points_json, f, indent=4)

def visualize_points(img: np.ndarray, points: List[tuple[int, int]]):
    vis_img = img.copy()
    for p in points:
        cv2.circle(vis_img, (p["x"], p["y"]), 2, (0, 0, 255), -1)
    return vis_img.astype(np.uint8)


if __name__ == "__main__":
    main()
