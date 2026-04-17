import cv2


def detect_sift_keypoints(gray_image):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray_image, None)
    return keypoints


def draw_sift_keypoints(image_bgr, keypoints):
    output = cv2.drawKeypoints(
        image_bgr,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return output


def count_keypoints(keypoints):
    return len(keypoints)