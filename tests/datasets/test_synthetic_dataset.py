from typing import List

import numpy as np
import pytest
from omegaconf import OmegaConf

from superpanopoint.datasets import DataSample
from superpanopoint.datasets.synthetic import SyntheticDataset

cfg = OmegaConf.load("tests/config/config_synthetic.yaml")

@pytest.mark.parametrize("data_samples", [
    [DataSample("data/synthetic/perspective/imgs/0.png", "data/synthetic/perspective/points/0.json")],
    [DataSample("data/synthetic/perspective/imgs/1.png", "data/synthetic/perspective/points/1.json")]
])
def test_synthetic_dataset(data_samples: List[DataSample]):
    synthetic_dataset = SyntheticDataset(data_samples, False, **cfg.dataset.train)
    img, points = synthetic_dataset[0]
    c, h, w = img.shape
    assert img.shape == (1, h, w)
    assert points.shape == (h//8, w//8)

    points_flattened = np.zeros((h, w))
    for r in range(h//8):
        for c in range(w//8):
            if points[r, c] < 64:
                cur_points = np.zeros((64))
                cur_points[points[r, c]] = 1
                cur_points = cur_points.reshape((8, 8))
                points_flattened[r*8:(r+1)*8, c*8:(c+1)*8] = cur_points

    assert points_flattened.shape == (h, w)
    assert data_samples[0].load_points()[:,:,0].shape == (h, w)
    assert np.sum(points_flattened - data_samples[0].load_points()[:,:,0]) == 0
