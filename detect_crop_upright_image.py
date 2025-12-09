#!/usr/bin/env python3
"""
detect_and_upright.py
- Detect tags in a single image and save crops
- Optionally call orient_and_fix.py to make crops upright

Usage (example):
python scripts/detect_and_upright.py \
    --model models/detect/best.pt \
    --image "inputs/images/your.jpg" \
    --out outputs/crops \
    --upright_out outputs/upright \
    --upright_model models/upright/best.pt \
    --conf 0.1

If you want to skip the upright step:
    --no_upright
"""
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import subprocess
import sys
import os

def crop_from_image(model_path, image_path, output_folder, conf=0.2, save_annotated=False):
    # Load model
    print(f"[detect] Loading detector: {model_path}")
    model = YOLO(str(model_path))

    # Prepare paths
    image_path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Run YOLO prediction
    print("[detect] Running detection...")
    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    n = 0
    if not len(result.boxes):
        print("[detect] No detections found.")
        return []

    saved_paths = []
    print(f"[detect] Found {len(result.boxes)} detections. Saving crops...")
    for i, box in enumerate(result.boxes):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        except Exception:
            # fallback if structure slightly different
            coords = box.xyxy[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        # clip to image
        h, w = img.shape[:2]
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # filename: originalname_crop_0001.jpg
        base = image_path.stem
        crop_fname = out_dir / f"{base}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_fname), crop)
        saved_paths.append(str(crop_fname))
        n += 1
        print(f"[detect] Saved: {crop_fname}")

    if save_annotated:
        # write an annotated image next to crops for debugging
        annot = img.copy()
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annot, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annot, f"{i+1}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        annot_path = out_dir / f"{base}_annotated.jpg"
        cv2.imwrite(str(annot_path), annot)
        print(f"[detect] Annotated image saved: {annot_path}")

    print(f"[detect] Done. Saved {n} crops to {out_dir}")
    return saved_paths

def call_orient_script(orient_script_path, weights_path, source_folder, output_folder, device="cpu", accept_conf=0.35):
    """
    Calls your orient_and_fix.py script via subprocess.
    Expects orient_and_fix.py to accept:
      --weights <weights> --source <source_folder> --output <output_folder> --device <device> --accept-conf <conf>
    """
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
    print("[orient] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Orientation script failed (rc={proc.returncode})")
    print(f"[orient] Completed. Upright images should be in: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Detect tags, crop them, then optionally make crops upright.")
    parser.add_argument("--model", required=True, help="path to tag detector weights (e.g. models/detect/best.pt)")
    parser.add_argument("--image", required=True, help="path to test image")
    parser.add_argument("--out", default="outputs/crops", help="where to save crops")
    parser.add_argument("--conf", type=float, default=0.1, help="confidence threshold for detection")
    parser.add_argument("--save_annotated", action="store_true", help="save annotated image with boxes")
    # upright options
    parser.add_argument("--upright", dest="upright", action="store_true", help="run upright step (calls orient_and_fix.py)")
    parser.add_argument("--no_upright", dest="upright", action="store_false", help="skip upright step")
    parser.set_defaults(upright=True)
    parser.add_argument("--upright_model", default="models/upright/best.pt", help="orientation model weights for orient_and_fix.py")
    parser.add_argument("--upright_out", default="outputs/upright", help="where to save upright images")
    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="path to your orient_and_fix.py")
    parser.add_argument("--device", default="cpu", help="device string to pass to orient script (cpu or cuda:0)")
    parser.add_argument("--accept_conf", type=float, default=0.35, help="accept-conf for orientation script")

    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    out_dir = Path(args.out)
    upright_out = Path(args.upright_out)

    # run detection + cropping
    crops = crop_from_image(model_path, image_path, out_dir, conf=args.conf, save_annotated=args.save_annotated)

    if not crops:
        print("[main] No crops produced. Exiting.")
        return

    if args.upright:
        # ensure upright output folder exists
        upright_out.mkdir(parents=True, exist_ok=True)
        # call the external orient script
        call_orient_script(args.orient_script, args.upright_model, out_dir, upright_out, device=args.device, accept_conf=args.accept_conf)
    else:
        print("[main] Upright step skipped (--no_upright).")

    print("[main] All done.")

if __name__ == "__main__":
    main()

