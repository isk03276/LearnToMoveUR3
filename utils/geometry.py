import math

import numpy as np


def get_distance_between_two_pts(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """
    Calculate distance between two points.
    Args:
        pts1 (list): first point
        pts2 (list): second point

    Returns:
        flot: distance
    """
    return np.linalg.norm(pts1 - pts2)
