from typing import Tuple

import numpy as np

from .utils import (InterpolationMode, interpolation_mode2order,
                    sample_equirec, uv2coor, xyz2uv, xyzpers)


def e2p(e_img: np.ndarray, 
        fov_deg: float, 
        u_deg: float, 
        v_deg: float, 
        out_hw: Tuple[int, int], 
        in_rot_deg: float=0, 
        mode: InterpolationMode='bilinear')->np.ndarray:
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    if mode not in interpolation_mode2order:
        raise NotImplementedError('unknown interpolation mode')

    if len(e_img.shape) == 2:
        e_img = e_img[..., np.newaxis]
    h, w = e_img.shape[:2]

    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    in_rot = in_rot_deg * np.pi / 180

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    pers_img = np.stack([
        sample_equirec(e_img[..., i], coor_xy, order=interpolation_mode2order[mode])
        for i in range(e_img.shape[2])
    ], axis=-1)

    return pers_img
