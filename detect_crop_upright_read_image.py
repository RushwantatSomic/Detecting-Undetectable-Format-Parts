#!/usr/bin/env python3
"""
detect_crop_upright_digits.py
Pipeline for a SINGLE IMAGE:
1. Detect tags -> save crops
2. Upright crops -> save upright
3. Run digit YOLO model -> save detections in JSON

Usage example:
python scripts/detect_crop_upright_digits.py \
  --model models/detect/best.pt \
  --image inputs/images/test.jpg \
  --upright_model models/upright/best.pt \
  --digit_model models/read/best.pt
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import subprocess
import sys


# ----------------------------------------------------------
# STEP 1 — DETECT & CROP
# ----------------------------------------------------------

def crop_from_image(model_path, image_path, output_folder, conf=0.2, save_annotated=False):
    print(f"[detect] Loading detector: {model_path}")
    model = YOLO(str(model_path))

    image_path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print("[detect] Running detection...")
    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if not len(result.boxes):
        print("[detect] No detections found.")
        return []

    saved_paths = []
    print(f"[detect] Found {len(result.boxes)} detections. Saving crops...")
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_filename = out_dir / f"{image_path.stem}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_filename), crop)
        saved_paths.append(str(crop_filename))
        print(f"[detect] Saved: {crop_filename}")

    print("[detect] Cropping complete.")
    return saved_paths


# ----------------------------------------------------------
# STEP 2 — ORIENT & UPRIGHT
# ----------------------------------------------------------

def call_orient_script(orient_script_path, weights_path, source_folder, output_folder, device="cpu", accept_conf=0.35):
    orient_script = Path(orient_script_path)
    if not orient_script.exists():
        raise FileNotFoundError(f"Orientation script not found: {orient_script}")

    cmd = [
        sys.executable,
        str(orient_script),
        "--weights", str(weights_path),
        "--source", str(source_folder),
        "--output", str(output_folder),
        "--device", str(device),
        "--accept-conf", str(accept_conf)
    ]

    print("\n[orient] Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError("Orientation script failed.")

    print("[orient] Orientation complete.")


# ----------------------------------------------------------
# STEP 3 — DIGIT DETECTION (trainWithSizing.py --mode infer)
# ----------------------------------------------------------

def call_digit_infer(digit_script_path, weights_path, source_folder, output_folder, conf=0.35):
    digit_script = Path(digit_script_path)
    if not digit_script.exists():
        raise FileNotFoundError(f"Digit script not found: {digit_script}")

    cmd = [
        sys.executable,
        str(digit_script_path),
        "--mode", "infer",
        "--weights", str(weights_path),
        "--input_folder", str(source_folder),
        "--out_folder", str(output_folder),
        "--conf", str(conf)
    ]

    print("\n[digits] Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError("Digit inference script failed.")

    print("[digits] Digit detection complete.")


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect tags → crop → upright → digit detect (single image)")
    parser.add_argument("--model", required=True, help="Path to TAG detector model")
    parser.add_argument("--image", required=True, help="Image to process")

    parser.add_argument("--out", default="outputs/crops", help="Crop output folder")
    parser.add_argument("--upright_out", default="outputs/upright", help="Upright output folder")
    parser.add_argument("--digits_out", default="outputs/digits", help="Digit JSON output folder")

    parser.add_argument("--upright_model", default="models/upright/best.pt", help="Upright model weights")
    parser.add_argument("--digit_model", default="models/read/best.pt", help="Digit detection model weights")

    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="Path to orient script")
    parser.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="Path to digit model script")

    parser.add_argument("--conf", type=float, default=0.1, help="Confidence for detection")
    parser.add_argument("--device", default="cpu", help="Device for upright model")

    args = parser.parse_args()

    # Step 1 — detect & crop
    crops = crop_from_image(args.model, args.image, args.out, conf=args.conf)
    if not crops:
        print("[main] No crops produced. Stopping.")
        return

    # Step 2 — upright
    call_orient_script(args.orient_script, args.upright_model, args.out, args.upright_out, device=args.device, accept_conf=args.conf)

    # Step 3 — read digits
    Path(args.digits_out).mkdir(parents=True, exist_ok=True)
    call_digit_infer(args.digit_script, args.digit_model, args.upright_out, args.digits_out, conf=args.conf)

    print("\n[main] 🎉 Pipeline complete!")
    print(" Crops saved to:     ", args.out)
    print(" Upright saved to:   ", args.upright_out)
    print(" Digit results saved:", args.digits_out)


if __name__ == "__main__":
    main()
