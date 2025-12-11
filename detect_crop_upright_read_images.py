#!/usr/bin/env python3
"""
detect_crop_upright_read_images.py
Batch image pipeline (process an entire folder of images):

For each image in --images_folder:
 1. Run tag detector -> write crops into <out>/<image_stem>/
 2. Call orient script on that per-image crop folder -> upright images saved under <upright_out>/<image_stem>/ (expected)
 3. Call digit inference on upright folder -> write digit outputs to <digits_out>/<image_stem>/
 4. Merge and deduplicate only this image's digit outputs into <digits_out>/<image_stem>/all_digits.txt

This script keeps image-run outputs separate from any video-run outputs by using per-image subfolders.

Usage example:
python scripts/detect_crop_upright_read_images.py \
  --model models/detect/best.pt \
  --images_folder inputs/images \
  --upright_model models/upright/best.pt \
  --digit_model models/read/best.pt
"""

import argparse
import sys
import subprocess
import json
import re
from pathlib import Path
from collections import OrderedDict

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import cv2


# -----------------------
# Merge helper (dedupe + sort by _crop_XXXX)
# -----------------------
def merge_digit_outputs(digits_root_path, merged_output_path, image_stem=None):
    """
    Merge per-crop digit outputs (deduplicated) for a specific image stem.

    digits_root_path: folder to recursively search for .txt/.json files
    merged_output_path: path (file) to write final lines: "<tag_filename> : <digits...>"
    image_stem: optional string; if provided only merge entries whose image name starts with this stem.
    """
    digits_root = Path(digits_root_path)
    merged_path = Path(merged_output_path)

    entries = OrderedDict()  # img_name -> digits_str (keeps last-seen by default)

    def store(img_name, digits_str):
        if image_stem and not str(img_name).startswith(image_stem):
            return
        img_name = str(img_name).strip()
        entries[img_name] = digits_str

    # Prefer TXT files
    txt_files = sorted(digits_root.rglob("*.txt"))
    for t in txt_files:
        try:
            txt = t.read_text(encoding="utf-8").strip().splitlines()
            img_line = next((l for l in txt if l.lower().startswith("image:")), None)
            digits_line = next((l for l in txt if l.lower().startswith("digits:")), None)
            if img_line and digits_line:
                img_name = img_line.split(":", 1)[1].strip()
                digits = digits_line.split(":", 1)[1].strip()
                store(img_name, digits)
            else:
                # fallback: use file stem or full content as name
                candidate_name = t.stem
                content = " ".join([l.strip() for l in txt if l.strip()])
                m = re.search(r"image[:]\\s*(\S+)", content, flags=re.I)
                if m:
                    candidate_name = m.group(1).strip()
                store(candidate_name, content)
        except Exception:
            continue

    # Fallback to JSON files
    json_files = sorted(digits_root.rglob("*.json"))
    for j in json_files:
        try:
            raw = j.read_text(encoding="utf-8")
            data = json.loads(raw)
            img_name = data.get("image") or data.get("filename") or j.stem
            dets = data.get("detections", [])
            dets_sorted = sorted(dets, key=lambda d: (d.get("bbox")[0] if d.get("bbox") else 0) if d.get("bbox") else 0)
            digits = " ".join(str(int(d.get("class"))) for d in dets_sorted if "class" in d)
            store(str(img_name), digits)
        except Exception:
            continue

    if not entries:
        print(f"[merge] No digit outputs found under {digits_root} for image_stem={image_stem}")
        return None

    # Sort by crop index if names contain '_crop_XXXX' pattern
    def crop_index_key(name):
        m = re.search(r"_crop_(\d{1,6})", name)
        if m:
            return int(m.group(1))
        return 10**9

    with_index = [(n, entries[n]) for n in entries if re.search(r"_crop_\d+", n)]
    without_index = [(n, entries[n]) for n in entries if not re.search(r"_crop_\d+", n)]

    with_index.sort(key=lambda x: crop_index_key(x[0]))
    ordered = with_index + without_index

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text("\n".join(f"{n} : {v}" for n, v in ordered), encoding="utf-8")
    print(f"[merge] Merged {len(ordered)} entries -> {merged_path}")
    return merged_path


# ----------------------------------------------------------
# STEP 1 — DETECT & CROP (per-image subfolder)
# ----------------------------------------------------------
def crop_from_image_model(model, image_path, output_folder, conf=0.2):
    """
    Runs tag detection on a single image and writes crops into:
        <output_folder>/<image_stem>/
    Returns tuple: (Path(to per-image crop folder), list_of_saved_crop_paths)
    """
    image_path = Path(image_path)
    out_dir = Path(output_folder)
    image_out_dir = out_dir / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"[detect] Running detection on {image_path.name}...")
    # model here is an ultralytics YOLO instance
    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if not getattr(result, "boxes", None) or len(result.boxes) == 0:
        print(f"[detect] No detections found for {image_path.name}.")
        return image_out_dir, []

    saved_paths = []
    for i, box in enumerate(result.boxes):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        except Exception:
            coords = box.xyxy[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_filename = image_out_dir / f"{image_path.stem}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_filename), crop)
        saved_paths.append(str(crop_filename))

    print(f"[detect] Saved {len(saved_paths)} crops to {image_out_dir}")
    return image_out_dir, saved_paths


# ----------------------------------------------------------
# STEP 2 & 3 — ORIENT & DIGIT wrappers (call external scripts)
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

    print(f"\n[orient] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("Orientation script failed.")
    print("[orient] Orientation complete.")


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

    print(f"\n[digits] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("Digit inference script failed.")
    print("[digits] Digit detection complete.")


# ----------------------------------------------------------
# MAIN BATCH PIPELINE
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch images: detect tags → crop → upright → digit detect (per-image outputs)")
    parser.add_argument("--model", required=True, help="Path to TAG detector model")
    parser.add_argument("--images_folder", required=True, help="Folder of images to process")

    parser.add_argument("--out", default="outputs/crops", help="Crop output base folder (per-image subfolders created)")
    parser.add_argument("--upright_out", default="outputs/upright", help="Upright output base folder")
    parser.add_argument("--digits_out", default="outputs/digits", help="Digit JSON output base folder")

    parser.add_argument("--upright_model", default="models/upright/best.pt", help="Upright model weights")
    parser.add_argument("--digit_model", default="models/read/best.pt", help="Digit detection model weights")

    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="Path to orient script")
    parser.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="Path to digit model script")

    parser.add_argument("--conf", type=float, default=0.1, help="Confidence for detection")
    parser.add_argument("--device", default="cpu", help="Device for upright model")

    args = parser.parse_args()

    images_folder = Path(args.images_folder)
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # image extensions to process
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [p for p in sorted(images_folder.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    if not images:
        print(f"[main] No images found in {images_folder} (extensions: {sorted(exts)})")
        return

    # Load detector model once
    if YOLO is None:
        raise RuntimeError("ultralytics.YOLO not available. Install ultralytics and try again.")

    detector = YOLO(str(args.model))
    print(f"[main] Loaded detector: {args.model}")

    for img_path in images:
        print("\n=========================================================")
        print(f"[main] Processing image: {img_path.name}")
        try:
            # Step 1: detect & crop (per-image folder)
            crop_folder, crops = crop_from_image_model(detector, img_path, args.out, conf=args.conf)
            if not crops:
                print(f"[main] No crops for {img_path.name}, skipping upright/digit steps.")
                continue

            # Step 2: orient — point at only this image's crop folder, expect oriented outputs under upright_out/<image_stem>/
            call_orient_script(args.orient_script, args.upright_model, crop_folder, args.upright_out, device=args.device, accept_conf=args.conf)

            # Determine upright source folder for digit inference
            expected_upright_image_folder = Path(args.upright_out) / img_path.stem
            if expected_upright_image_folder.exists() and any(expected_upright_image_folder.iterdir()):
                digit_source_folder = expected_upright_image_folder
            else:
                # fallback: sometimes orient script writes upright images directly under upright_out with same filenames
                base_upright = Path(args.upright_out)
                found_match = False
                for c in crops[:5]:
                    if (base_upright / Path(c).name).exists():
                        found_match = True
                        break
                digit_source_folder = base_upright if found_match else crop_folder

            # Step 3: call digit inference writing to per-image digits folder
            per_image_digits_out = Path(args.digits_out) / img_path.stem
            per_image_digits_out.mkdir(parents=True, exist_ok=True)
            print(f"[main] Running digit inference for {img_path.name} on: {digit_source_folder}")
            call_digit_infer(args.digit_script, args.digit_model, digit_source_folder, per_image_digits_out, conf=args.conf)

            # Step 4: merge only this image's outputs
            final_digits = per_image_digits_out / "all_digits.txt"
            merge_digit_outputs(per_image_digits_out, final_digits, image_stem=img_path.stem)

            print(f"[main] Completed processing for {img_path.name}")

        except Exception as e:
            print(f"[main] Error processing {img_path.name}: {e}")
            continue

    print("\n[main] 🎉 Batch image pipeline complete!")
    print(f" Crops base folder:   {args.out}")
    print(f" Upright base folder: {args.upright_out}")
    print(f" Digits base folder:  {args.digits_out}")


if __name__ == "__main__":
    main()
