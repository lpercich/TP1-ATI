import cv2
import numpy as np


def sobel_edges(gray_image, ksize: int = 3, blur_ksize: int = 5, threshold_ratio: float = 0.25):
    if ksize not in (3, 5, 7):
        raise ValueError("ksize debe ser 3, 5 o 7")

    if blur_ksize > 0:
        gray_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    magnitude_uint8 = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    threshold_value = int(threshold_ratio * magnitude_uint8.max())
    _, edge_mask = cv2.threshold(
        magnitude_uint8, threshold_value, 255, cv2.THRESH_BINARY
    )

    return magnitude_uint8, edge_mask