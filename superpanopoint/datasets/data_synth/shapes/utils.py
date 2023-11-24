import cv2
import numpy as np

random_state = np.random.RandomState(None)


def angle_between_vectors(v1, v2):
    """Compute the angle (in rad) between the two vectors v1 and v2"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_random_color(background_color: int):
    """Output a random scalar in grayscale with at least a small contrast with the background color"""
    color = random_state.randint(256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def keep_points_inside(points: np.ndarray, img_w: int, img_h: int):
    """Keep only the points whose coordinates are inside the dimensions of the image"""
    mask = (points[:, 0] >= 0) & (points[:, 0] < img_w) &\
           (points[:, 1] >= 0) & (points[:, 1] < img_h)
    return points[mask, :]


def random_fillPoly(img: np.ndarray, points: np.ndarray, color: int):
    """Return a random fillPoly function"""
    if random_state.rand() > 0.3:
        return fillPoly_with_noise(img, points, color)
    else:
        return cv2.fillPoly(img, [points], color)


def fillPoly_with_noise(img: np.ndarray, points: np.ndarray, color: int):
    """Fill a polygon with a color and add some noise"""
    h, w = img.shape[:2]
    tmp_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tmp_img, [points], 1)
    rows, cols = np.where(tmp_img > 0)
    if len(rows) == 0:
        return img
    base_noise_img = generate_background(w, h, nb_blobs=100, min_rad_ratio=0.01, min_kernel_size=50, max_kernel_size=51).astype(int)
    mean_col = max(1, np.mean(base_noise_img[rows, cols]))
    base_noise_img = base_noise_img * (color / mean_col)
    base_noise_img = np.minimum(np.maximum(base_noise_img, 0), 255).astype(np.uint8)
    img[rows, cols] = base_noise_img[rows, cols]
    return img


def generate_background(img_w: int, img_h: int, nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=50, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    dim = max((img_h, img_w))
    cv2.randu(img, 0, 255)
    cv2.threshold(img, random_state.randint(256), 255, cv2.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate([random_state.randint(0, img_w, size=(nb_blobs, 1)),
                            random_state.randint(0, img_h, size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv2.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(int(dim * min_rad_ratio),
                                    int(dim * max_rad_ratio)),
                  col, -1)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv2.blur(img, (kernel_size, kernel_size), img)
    return img


def generate_symetric_background(img_w: int, img_h: int, nb_blobs=100, min_rad_ratio=0.03,
                        max_rad_ratio=0.05, min_kernel_size=100, max_kernel_size=300):
    if img_w % 2 == 1 or img_h % 2 == 1:
        raise ValueError("The image dimensions must be even")
    bg_img = generate_background(img_w//2, img_h//2, nb_blobs, min_rad_ratio, max_rad_ratio, min_kernel_size, max_kernel_size)
    v_flip = bg_img[::-1, :]
    h_flip = bg_img[:, ::-1]
    hv_flip = bg_img[::-1, ::-1]
    return np.concatenate([np.concatenate([bg_img, h_flip], axis=1), np.concatenate([v_flip, hv_flip], axis=1)], axis=0)
