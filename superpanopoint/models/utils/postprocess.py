import numpy as np


def non_maximum_suppression(pointness: np.ndarray, radius: int=4, strict=True) -> np.ndarray:
    """
    Args:
        pointness (np.ndarray): probability map of pointness. (h, w)
        radius (int, optional): radus of non-maximum suppression. Defaults to 4.
        strict (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: binary point mask after non-maximum suppression. (h, w)
    """
    non_zero_ys, non_zero_xs = np.nonzero(pointness)
    for x, y in zip(non_zero_xs, non_zero_ys):
        if pointness[y, x] == 0:
            continue
        else:
            orig_pointness = pointness[y, x]
            area_xs = np.arange(max(0, x - radius), min(pointness.shape[1], x + radius + 1))
            area_ys = np.arange(max(0, y - radius), min(pointness.shape[0], y + radius + 1))
            mesh_xs, mesh_ys = np.meshgrid(area_xs, area_ys)
            mesh_xs = mesh_xs.flatten()
            mesh_ys = mesh_ys.flatten()
            if strict:
                [target_idxs] = np.where(pointness[mesh_ys, mesh_xs] <= orig_pointness)
                pointness[mesh_ys[target_idxs], mesh_xs[target_idxs]] = 0
                pointness[y, x] = orig_pointness
            else:
                [target_idxs] = np.where(pointness[mesh_ys, mesh_xs] < orig_pointness)
                pointness[mesh_ys[target_idxs], mesh_xs[target_idxs]] = 0
    return pointness.astype(bool).astype(int)
