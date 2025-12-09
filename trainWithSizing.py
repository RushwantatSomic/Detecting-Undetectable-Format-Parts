#!/usr/bin/env python3
"""
trainWithSizing.py

- Original training routine (main) retained.
- Added inference mode (--mode infer) that runs YOLO inference on a folder,
  writes a per-image JSON with detections and an inference_summary.json.
- Includes optional size-filter helpers for post-processing / visualization.

Usage:
  # Train (original behavior)
  python trainWithSizing.py --mode train

  # Infer on folder of images (upright crops)
  python trainWithSizing.py --mode infer --weights models/read/best.pt --input_folder outputs/upright --out_folder outputs/digits --conf 0.1
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------------------
# Configuration for size filtering
# ---------------------------
CONFIDENCE_THRESHOLD = 0.50  # default for size-filtering visualization
SIZE_RATIO_THRESHOLD = 0.50  # relative threshold: keep boxes >= 50% of max height
MIN_PIXEL_HEIGHT = 15        # absolute min height in pixels

# ---------------------------
# Inference: run on folder and save results
# ---------------------------
def run_infer_on_folder(weights, input_folder, out_folder, conf=0.5):
    """
    Run YOLO inference on all images in a folder.
    Saves BOTH:
      - per-image JSON with full detection info
      - per-image TXT with only detected digits (sorted left→right)
    """
    model = YOLO(weights)
    input_folder = Path(input_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        p for p in input_folder.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ])

    results_summary = []

    for p in img_files:
        res_list = model.predict(str(p), conf=conf, verbose=False)
        if not res_list:
            continue

        r = res_list[0]
        detections = []

        # Collect detections with box + class + conf
        for b in getattr(r, "boxes", []):
            try:
                xy = b.xyxy[0].tolist()
                conf_score = float(b.conf[0])
                cls = int(b.cls[0])
            except Exception:
                continue

            detections.append({
                "bbox": [float(v) for v in xy],
                "conf": conf_score,
                "class": cls
            })

        # ---------- SAVE JSON ----------
        json_path = out_folder / (p.stem + ".json")
        with open(json_path, "w") as f:
            json.dump({"image": str(p), "detections": detections}, f, indent=2)

        # ---------- SAVE TXT (DIGITS ONLY) ----------

        # Sort detections by x1 position (left to right)
        sorted_dets = sorted(detections, key=lambda d: d["bbox"][0])

        digits_only = [str(det["class"]) for det in sorted_dets]

        txt_path = out_folder / (p.stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("image: " + p.name + "\n")
            f.write("digits: " + " ".join(digits_only) + "\n")

        print(f"Inferred {p.name} -> digits: {digits_only}")

        results_summary.append({
            "image": str(p),
            "n_detections": len(detections),
            "json": str(json_path),
            "txt": str(txt_path)
        })

    # Write summary
    with open(out_folder / "inference_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("Inference complete. Outputs saved to:", out_folder)



# ---------------------------
# Size filtering helper (works on ultralytics result object)
# ---------------------------
def filter_by_size_from_result(
    result,
    size_ratio_threshold: float = SIZE_RATIO_THRESHOLD,
    min_height: int = MIN_PIXEL_HEIGHT,
):
    """
    Filter detections in a single YOLO 'result' object based on bounding box height.

    Steps:
      1. Remove boxes whose height < min_height (absolute filter)
      2. Compute max height among remaining boxes
      3. Keep only boxes whose height >= size_ratio_threshold * max_height
    """
    boxes = result.boxes  # ultralytics.yolo.engine.results.Boxes

    # No detections
    if boxes is None or len(boxes) == 0:
        return result

    # xyxy shape: [N, 4]
    xyxy = boxes.xyxy  # (x1, y1, x2, y2)

    # Compute heights (tensor)
    try:
        heights = xyxy[:, 3] - xyxy[:, 1]
    except Exception:
        # If shape is unexpected, return as-is
        return result

    # 1) Absolute height filter
    keep_mask = heights >= min_height

    # If everything is filtered out
    if keep_mask.sum() == 0:
        # set to empty Boxes (safe to assign same type)
        result.boxes = boxes[keep_mask]
        return result

    # 2) Relative to max height among remaining
    max_h = heights[keep_mask].max()
    min_acceptable = max_h * size_ratio_threshold

    keep_mask = heights >= min_acceptable

    # Apply mask to boxes (Boxes supports indexing)
    result.boxes = boxes[keep_mask]

    return result


# ---------------------------
# Optional: inference with size filter + save annotated images
# ---------------------------
def run_inference_with_size_filter(
    weights,
    source,
    output_folder: str = "inference_with_size_filter",
    conf: float = CONFIDENCE_THRESHOLD,
    size_ratio_threshold: float = SIZE_RATIO_THRESHOLD,
    min_height: int = MIN_PIXEL_HEIGHT,
):
    """
    Run inference and save annotated images that show only the boxes after size filtering.
    """
    model = YOLO(weights)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run YOLO inference
    results = model(source, conf=conf)

    # Handle single or multiple results
    if not isinstance(results, list):
        results = [results]

    for i, res in enumerate(results):
        # Apply size-based filtering to this result
        res_filtered = filter_by_size_from_result(res, size_ratio_threshold=size_ratio_threshold, min_height=min_height)

        # Draw only the filtered boxes
        try:
            annotated = res_filtered.plot()
        except Exception:
            # fallback: use original res.plot()
            annotated = res.plot()

        # Build output path
        img_name = f"filtered_result_{i}.jpg"
        out_path = out_dir / img_name

        # Save annotated image (res.plot() returns numpy array BGR)
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved filtered result to: {out_path}")


# ---------------------------
# ORIGINAL TRAINING (unchanged)
# ---------------------------
def main_train():
    """
    Original training routine. Adjust parameters as needed.
    """
    # Load a pretrained YOLOv8 model (choose nano/small/medium/etc.)
    model = YOLO("yolov8n.pt")  # change if you want a different base model

    # Train the model
    model.train(
        data="data.yaml",   # path to your data.yaml
        epochs=100,         # adjust as needed
        imgsz=640,          # training image size
        batch=16,           # adjust depending on your GPU
        workers=8,
        device="cuda" if YOLO().device and "cuda" in str(YOLO().device) else "cpu"
    )

    # Optional: evaluate on test set (standard validation)
    try:
        model.val()
    except Exception:
        pass

    # Note: you can optionally call run_inference_with_size_filter(...) here for quick checks.


# ---------------------------
# CLI entrypoint: train or infer
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainWithSizing: train or infer with size helpers")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="train or infer")
    parser.add_argument("--weights", default="yolov8n.pt", help="weights for inference or base for training")
    parser.add_argument("--input_folder", default=None, help="folder of images for inference")
    parser.add_argument("--out_folder", default="results_infer", help="where to write inference jsons")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold for inference")
    # Optional args for size filter visualization
    parser.add_argument("--run_size_filter_viz", action="store_true", help="run inference with size filter visualization (saves annotated images)")
    parser.add_argument("--size_viz_out", default="inference_with_size_filter", help="where to save size-filter annotated images")
    parser.add_argument("--size_ratio_threshold", type=float, default=SIZE_RATIO_THRESHOLD, help="size ratio threshold for visualization")
    parser.add_argument("--min_height", type=int, default=MIN_PIXEL_HEIGHT, help="min pixel height for visualization")
    args = parser.parse_args()

    if args.mode == "train":
        main_train()
    else:
        if args.input_folder is None:
            raise SystemExit("Infer mode requires --input_folder")
        # Run inference and save JSONs
        run_infer_on_folder(args.weights, args.input_folder, args.out_folder, conf=args.conf)

        # Optionally run visualization with size filtering
        if args.run_size_filter_viz:
            run_inference_with_size_filter(
                args.weights,
                args.input_folder,
                output_folder=args.size_viz_out,
                conf=args.conf,
                size_ratio_threshold=args.size_ratio_threshold,
                min_height=args.min_height,
            )
