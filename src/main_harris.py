import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from utils import ensure_dir, load_image_gray, load_image_bgr, save_image
from harris import detect_harris_corners, evaluate_stability

def run_harris_experiment():
    data_dir = Path("data/images")
    results_dir = Path("results/harris")
    ensure_dir(results_dir)

    image_paths = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
    if not image_paths:
        print(f"No se encontraron imágenes en {data_dir}")
        return

    k_values = [0.04, 0.06]
    window_sizes = [3, 5]
    
    dist_tol = 3.0

    all_metrics = []

    for img_path in image_paths:
        img_name = img_path.stem
        print(f"Procesando {img_name}...")

        img_bgr = load_image_bgr(img_path)
        img_gray = load_image_gray(img_path)
        total_pixels = img_gray.size

        for k in k_values:
            for w in window_sizes:
                print(f"  k={k}, window_size={w}")

                corners, dst, threshold = detect_harris_corners(img_gray, window_size=w, k=k)
                
                total_corners = len(corners)
                density = total_corners / total_pixels

                stab_pos15 = evaluate_stability(img_gray, corners, angle=15.0, window_size=w, k=k, distance_tolerance=dist_tol)
                stab_neg15 = evaluate_stability(img_gray, corners, angle=-15.0, window_size=w, k=k, distance_tolerance=dist_tol)
                
                stability_avg = (stab_pos15 + stab_neg15) / 2.0

                all_metrics.append({
                    "image": img_name,
                    "k": k,
                    "window_size": w,
                    "total_corners": total_corners,
                    "density": density,
                    "stability_pos15_pct": stab_pos15,
                    "stability_neg15_pct": stab_neg15,
                    "stability_avg_pct": stability_avg
                })

                img_out = img_bgr.copy()
                mask = (dst > threshold).astype(np.uint8) * 255
                mask_dilated = cv2.dilate(mask, None)
                
                img_out[mask_dilated == 255] = [0, 0, 255]
                
                out_filename = results_dir / f"{img_name}_harris_k{k}_w{w}_corners.png"
                save_image(out_filename, img_out)

    df = pd.DataFrame(all_metrics)
    csv_path = results_dir / "harris_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMétricas guardadas en {csv_path}")

    summary_df = df.groupby(["k", "window_size"]).agg({
        "total_corners": ["mean", "std"],
        "density": ["mean", "std"],
        "stability_avg_pct": ["mean", "std"]
    }).reset_index()
    
    summary_csv_path = results_dir / "harris_summary_by_params.csv"
    summary_df.to_csv(summary_csv_path)
    print(f"Resumen guardado en {summary_csv_path}")

if __name__ == "__main__":
    run_harris_experiment()