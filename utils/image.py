import cv2
import numpy as np

def resize_image(image:np.ndarray, width:int, height:int)-> np.ndarray:
    """
    Resize a image.
    Args:
        image (np.ndarray): input image
        width (int): width size to resize
        height (int): height size to resize

    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, dsize=(width, height))