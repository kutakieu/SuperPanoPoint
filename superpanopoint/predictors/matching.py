import numpy as np


def calc_matching(kps1: np.ndarray, desc1: np.ndarray, kps2: np.ndarray, desc2: np.ndarray, threshold: float=0.8):
    ele_wise_dots = np.einsum("nc,mc->nm", desc1, desc2)
    best_matches = np.argmax(ele_wise_dots, axis=1)
    best_match_dots = np.max(ele_wise_dots, axis=1)
    valid_matches = best_match_dots > threshold
    
