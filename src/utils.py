from pathlib import Path
from typing import Union
import cv2
import numpy as np


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_bgr(image_path: Union[str, Path]):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def load_image_gray(image_path: Union[str, Path]):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def save_image(image_path: Union[str, Path], image) -> None:
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen: {image_path}")


def to_binary_mask(image) -> np.ndarray:
    return (image > 0).astype(np.uint8)


def find_ground_truth_path(gt_dir: Union[str, Path], image_path: Union[str, Path]) -> Path:
    gt_dir = Path(gt_dir)
    image_path = Path(image_path)
    stem = image_path.stem

    candidates = [
        gt_dir / f"{stem}.png",
        gt_dir / f"{stem}.jpg",
        gt_dir / f"{stem}.jpeg",
        gt_dir / f"{stem}.bmp",
        gt_dir / f"{stem}.tif",
        gt_dir / f"{stem}.tiff",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No se encontró ground truth para {image_path.name} en {gt_dir}"
    )