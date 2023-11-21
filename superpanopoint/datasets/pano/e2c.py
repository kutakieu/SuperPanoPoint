import numpy as np

from .utils import (CubeFormat, InterpolationMode, horizon_cube2cube_func,
                    interpolation_mode2order, sample_equirec, uv2coor, xyz2uv,
                    xyzcube)


def e2c(e_img: np.ndarray, 
        face_w: int=256, 
        mode: InterpolationMode='bilinear', 
        cube_format: CubeFormat='dice')->np.ndarray:
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    if mode not in interpolation_mode2order:
        raise NotImplementedError('unknown interpolation mode')
    if cube_format not in horizon_cube2cube_func:
        raise NotImplementedError('unknown cube map format')


    if len(e_img.shape) == 2:
        e_img = e_img[..., np.newaxis]
    h, w = e_img.shape[:2]

    xyz = xyzcube(face_w)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    cubemap = np.stack([
        sample_equirec(e_img[..., i], coor_xy, order=interpolation_mode2order[mode])
        for i in range(e_img.shape[2])
    ], axis=-1)

    cubemap = horizon_cube2cube_func[cube_format](cubemap)
    return cubemap
