import cv2
import numpy as np


def detect_harris_corners(image_gray, window_size: int, k: float, threshold_ratio: float = 0.01):
    gray = np.float32(image_gray)
    
    dst = cv2.cornerHarris(gray, window_size, ksize=3, k=k)
    
    threshold = threshold_ratio * dst.max()
    
    y_coords, x_coords = np.where(dst > threshold)
    corners = list(zip(x_coords, y_coords))
    
    return corners, dst, threshold


def rotate_image_and_matrix(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated, M


def evaluate_stability(image_gray, orig_corners, angle: float, window_size: int, k: float, threshold_ratio: float = 0.01, distance_tolerance: float = 3.0):
    if not orig_corners:
        return 0.0
        
    rotated_img, M = rotate_image_and_matrix(image_gray, angle)
    rot_corners, _, _ = detect_harris_corners(rotated_img, window_size, k, threshold_ratio)
    
    if not rot_corners:
        return 0.0
        
    rot_corners_arr = np.array(rot_corners)
    orig_corners_arr = np.array(orig_corners)
    ones = np.ones((orig_corners_arr.shape[0], 1))
    orig_corners_homo = np.hstack([orig_corners_arr, ones])
    
    transformed_corners = M.dot(orig_corners_homo.T).T
    
    maintained_count = 0
    for pt in transformed_corners:
        distances = np.linalg.norm(rot_corners_arr - pt, axis=1)
        if np.any(distances <= distance_tolerance):
            maintained_count += 1
            
    stability_percentage = (maintained_count / len(orig_corners)) * 100.0
    return stability_percentage