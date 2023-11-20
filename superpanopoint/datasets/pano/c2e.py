import numpy as np

from .utils import (CubeFormat, InterpolationMode, cube_format2horizontal_func,
                    equirect_facetype, equirect_uvgrid,
                    interpolation_mode2order, sample_cubefaces)


def c2e(cubemap, 
        equi_h: int, 
        equi_w: int, 
        mode: InterpolationMode='bilinear', 
        cube_format: CubeFormat='dice')->np.ndarray:

    if mode not in interpolation_mode2order:
        raise NotImplementedError('unknown interpolation mode')
    if cube_format not in cube_format2horizontal_func:
        raise NotImplementedError('unknown cube format')

    cubemap_horizon = cube_format2horizontal_func[cube_format](cubemap)
    
    assert cubemap_horizon.shape[0] * 6 == cubemap_horizon.shape[1]
    assert equi_w % 8 == 0
    
    if len(cubemap_horizon.shape) == 2:
        cubemap_horizon = cubemap_horizon[..., np.newaxis]

    face_w = cubemap_horizon.shape[0]

    uv = equirect_uvgrid(equi_h, equi_w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap_horizon, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(equi_h, equi_w)
    coor_x = np.zeros((equi_h, equi_w))
    coor_y = np.zeros((equi_h, equi_w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=interpolation_mode2order[mode])
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec
