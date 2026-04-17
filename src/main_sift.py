from pathlib import Path
import pandas as pd

from utils import ensure_dir, load_image_bgr, load_image_gray, save_image
from sift import detect_sift_keypoints, draw_sift_keypoints, count_keypoints


def run_sift_experiment():
    input_dir = Path("data/images")
    output_dir = Path("results/sift")

    ensure_dir(output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        raise RuntimeError("No se encontraron imágenes en data/images")

    rows = []

    for image_path in image_paths:
        image_bgr = load_image_bgr(image_path)
        gray = load_image_gray(image_path)

        keypoints = detect_sift_keypoints(gray)
        vis_image = draw_sift_keypoints(image_bgr, keypoints)
        num_keypoints = count_keypoints(keypoints)

        output_image_path = output_dir / f"{image_path.stem}_sift_keypoints.png"
        save_image(output_image_path, vis_image)

        rows.append({
            "image": image_path.name,
            "num_keypoints": num_keypoints
        })

        print(
            f"[OK] {image_path.name} | "
            f"SIFT keypoints = {num_keypoints}"
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "sift_metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nResultados guardados en: {output_dir}")
    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    run_sift_experiment()