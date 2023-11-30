import numpy as np
import pytest

from superpanopoint.datasets.data_synth import generate_perspective_sample


@pytest.mark.parametrize("img_w, img_h", [(512, 512), (1024, 1024), (1024, 512)])
def test_generate_perspective_sample(img_w, img_h):
    synth_data = generate_perspective_sample(img_w, img_h)
    assert synth_data.synth_img.shape == (img_h, img_w)

    points = synth_data.points_as_img()
    num_points = 0
    for shape in synth_data.added_shapes:
        num_points += len(shape.points)
    assert np.sum(points) == num_points
    assert np.sum(points) > 0
