from pathlib import Path
import pandas as pd

from utils import (
    ensure_dir,
    load_image_gray,
    save_image,
    find_ground_truth_path,
)
from edges import sobel_edges, canny_edges
from metrics import (
    edge_density,
    connected_components_info,
    average_component_length,
    precision_recall_f1,
)


def run_sobel_experiment():
    input_dir = Path("data/images")
    gt_dir = Path("data/ground_truth")
    output_dir = Path("results/sobel")

    ensure_dir(output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        raise RuntimeError("No se encontraron imágenes en data/images")

    sobel_kernel_sizes = [3, 5, 7]
    tolerance = 1

    rows = []

    for image_path in image_paths:
        gray = load_image_gray(image_path)

        try:
            gt_path = find_ground_truth_path(gt_dir, image_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        gt_mask = load_image_gray(gt_path)
        image_name = image_path.stem

        for ksize in sobel_kernel_sizes:
            magnitude, edge_mask = sobel_edges(
                gray_image=gray,
                ksize=ksize,
                blur_ksize=5,
                threshold_ratio=0.25
            )

            mag_path = output_dir / f"{image_name}_sobel_k{ksize}_magnitude.png"
            edge_path = output_dir / f"{image_name}_sobel_k{ksize}_edges.png"

            save_image(mag_path, magnitude)
            save_image(edge_path, edge_mask)

            density = edge_density(edge_mask)
            num_components, _ = connected_components_info(edge_mask)
            avg_length = average_component_length(edge_mask)
            precision, recall, f1 = precision_recall_f1(
                edge_mask, gt_mask, tolerance=tolerance
            )

            rows.append({
                "image": image_path.name,
                "ground_truth": gt_path.name,
                "method": "sobel",
                "ksize": ksize,
                "blur_ksize": 5,
                "threshold_ratio": 0.25,
                "tolerance_px": tolerance,
                "edge_density": density,
                "num_components": num_components,
                "avg_component_length": avg_length,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            })

            print(
                f"[OK] {image_path.name} | Sobel k={ksize} | "
                f"density={density:.4f} | "
                f"components={num_components} | "
                f"avg_length={avg_length:.2f} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"f1={f1:.4f}"
            )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "sobel_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary_df = (
        df.groupby("ksize", as_index=False)[
            [
                "edge_density",
                "num_components",
                "avg_component_length",
                "precision",
                "recall",
                "f1_score",
            ]
        ]
        .mean()
        .sort_values("ksize")
    )

    summary_csv_path = output_dir / "sobel_summary_by_ksize.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nResultados guardados en: {output_dir}")
    print(f"CSV detallado: {csv_path}")
    print(f"CSV resumen:  {summary_csv_path}")


def run_canny_experiment():
    input_dir = Path("data/images")
    gt_dir = Path("data/ground_truth")
    output_dir = Path("results/canny")

    ensure_dir(output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        raise RuntimeError("No se encontraron imágenes en data/images")

    low_thresholds = [50, 100]
    high_thresholds = [150, 200]
    tolerance = 1

    rows = []

    for image_path in image_paths:
        gray = load_image_gray(image_path)

        try:
            gt_path = find_ground_truth_path(gt_dir, image_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        gt_mask = load_image_gray(gt_path)
        image_name = image_path.stem

        for low_th in low_thresholds:
            for high_th in high_thresholds:
                edge_mask = canny_edges(
                    gray_image=gray,
                    low_threshold=low_th,
                    high_threshold=high_th,
                    blur_ksize=5
                )

                edge_path = output_dir / f"{image_name}_canny_l{low_th}_h{high_th}_edges.png"

                save_image(edge_path, edge_mask)

                density = edge_density(edge_mask)
                num_components, _ = connected_components_info(edge_mask)
                avg_length = average_component_length(edge_mask)
                precision, recall, f1 = precision_recall_f1(
                    edge_mask, gt_mask, tolerance=tolerance
                )

                rows.append({
                    "image": image_path.name,
                    "ground_truth": gt_path.name,
                    "method": "canny",
                    "low_threshold": low_th,
                    "high_threshold": high_th,
                    "blur_ksize": 5,
                    "tolerance_px": tolerance,
                    "edge_density": density,
                    "num_components": num_components,
                    "avg_component_length": avg_length,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                })

                print(
                    f"[OK] {image_path.name} | Canny L={low_th} H={high_th} | "
                    f"density={density:.4f} | "
                    f"components={num_components} | "
                    f"avg_length={avg_length:.2f} | "
                    f"precision={precision:.4f} | "
                    f"recall={recall:.4f} | "
                    f"f1={f1:.4f}"
                )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "canny_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary_df = (
        df.groupby(["low_threshold", "high_threshold"], as_index=False)[
            [
                "edge_density",
                "num_components",
                "avg_component_length",
                "precision",
                "recall",
                "f1_score",
            ]
        ]
        .mean()
        .sort_values(["low_threshold", "high_threshold"])
    )

    summary_csv_path = output_dir / "canny_summary_by_thresholds.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nResultados guardados en: {output_dir}")
    print(f"CSV detallado: {csv_path}")
    print(f"CSV resumen:  {summary_csv_path}")


if __name__ == "__main__":
    print("--- Iniciando Experimento Sobel ---")
    run_sobel_experiment()
    print("\n--- Iniciando Experimento Canny ---")
    run_canny_experiment()