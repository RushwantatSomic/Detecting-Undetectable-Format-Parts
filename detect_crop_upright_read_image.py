#!/usr/bin/env python3
"""
detect_crop_upright_digits.py
Pipeline for a SINGLE IMAGE:
1. Detect tags -> save crops (saved to per-image subfolder: <out>/<image_stem>/)
2. Upright crops -> save upright (we point orient script to the per-image crop folder)
3. Run digit YOLO model -> save detections in JSON/TXT
4. Merge all per-crop outputs for THIS IMAGE into a single text file

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
import json

# -----------------------
# Merge helper
# -----------------------
from collections import OrderedDict
import re

def merge_digit_outputs(digits_root_path, merged_output_path, image_stem=None):
    """
    Merge per-crop digit outputs (deduplicated).

    digits_root_path: folder to recursively search for .txt/.json files
    merged_output_path: path (file) to write final lines: "<tag_filename> : <digits...>"
    image_stem: optional string; if provided only merge entries whose image name starts with this stem.
    """
    digits_root = Path(digits_root_path)
    merged_path = Path(merged_output_path)

    # map img_name -> digits string (keeps last-seen by default)
    entries = OrderedDict()

    # helper to store entry (overwrites previous if same img_name seen again)
    def store(img_name, digits_str):
        if image_stem and not img_name.startswith(image_stem):
            return
        entries[img_name] = digits_str

    # first prefer .txt files
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
                # If content looks like "image: <name>", try to extract
                m = re.search(r"image[:]\s*(\S+)", content, flags=re.I)
                if m:
                    candidate_name = m.group(1).strip()
                store(candidate_name, content)
        except Exception:
            continue

    # fallback to JSON files if no txt entries found
    if not entries:
        json_files = sorted(digits_root.rglob("*.json"))
        for j in json_files:
            try:
                raw = j.read_text(encoding="utf-8")
                data = json.loads(raw)
                # try to find image name inside json
                img_name = data.get("image") or data.get("filename") or j.stem
                dets = data.get("detections", [])
                dets_sorted = sorted(dets, key=lambda d: (d.get("bbox")[0] if d.get("bbox") else 0))
                digits = " ".join(str(int(d.get("class"))) for d in dets_sorted if "class" in d)
                store(str(img_name), digits)
            except Exception:
                continue

    if not entries:
        print(f"[merge] No digit outputs found under {digits_root} for image_stem={image_stem}")
        return None

    # Sort by crop index if names contain '_crop_XXXX' pattern, else keep insertion order.
    def crop_index_key(name):
        m = re.search(r"_crop_(\d{1,6})", name)
        if m:
            return int(m.group(1))
        # fallback: try to preserve name order by returning large number plus hash
        return 10**9

    # create ordered list: first names with crop index sorted, then the rest in insertion order
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
def crop_from_image(model_path, image_path, output_folder, conf=0.2, save_annotated=False):
    """
    Runs tag detection on a single image and writes crops into:
        <output_folder>/<image_stem>/
    Returns tuple: (Path(to per-image crop folder), list_of_saved_crop_paths)
    """
    print(f"[detect] Loading detector: {model_path}")
    model = YOLO(str(model_path))

    image_path = Path(image_path)
    out_dir = Path(output_folder)
    # create a per-image subfolder to avoid mixing with other runs
    image_out_dir = out_dir / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print("[detect] Running detection...")
    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if not getattr(result, "boxes", None) or len(result.boxes) == 0:
        print("[detect] No detections found.")
        return image_out_dir, []  # return folder and empty list for compatibility

    saved_paths = []
    print(f"[detect] Found {len(result.boxes)} detections. Saving crops to {image_out_dir} ...")
    for i, box in enumerate(result.boxes):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        except Exception:
            coords = box.xyxy[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        # clamp coords
        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # simpler crop filename inside per-image subfolder
        crop_filename = image_out_dir / f"{image_path.stem}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_filename), crop)
        saved_paths.append(str(crop_filename))
        print(f"[detect] Saved: {crop_filename}")

    print("[detect] Cropping complete.")
    # return the folder path and saved file list so caller can use them
    return image_out_dir, saved_paths

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

    parser.add_argument("--out", default="outputs/crops", help="Crop output folder (base; per-image subfolder will be created inside)")
    parser.add_argument("--upright_out", default="outputs/upright", help="Upright output base folder")
    parser.add_argument("--digits_out", default="outputs/digits", help="Digit JSON output folder")

    parser.add_argument("--upright_model", default="models/upright/best.pt", help="Upright model weights")
    parser.add_argument("--digit_model", default="models/read/best.pt", help="Digit detection model weights")

    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="Path to orient script")
    parser.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="Path to digit model script")

    parser.add_argument("--conf", type=float, default=0.1, help="Confidence for detection")
    parser.add_argument("--device", default="cpu", help="Device for upright model")

    args = parser.parse_args()

    # Step 1 — detect & crop (returns per-image crop folder + saved paths)
    crop_folder, crops = crop_from_image(args.model, args.image, args.out, conf=args.conf)
    if not crops:
        print("[main] No crops produced. Stopping.")
        return

    # Step 2 — upright (point at only this image's crop folder)
    call_orient_script(args.orient_script, args.upright_model, crop_folder, args.upright_out, device=args.device, accept_conf=args.conf)

    # Step 3 — read digits (we try to pick the upright folder corresponding to this image)
    expected_upright_image_folder = Path(args.upright_out) / Path(args.image).stem
    if expected_upright_image_folder.exists() and any(expected_upright_image_folder.iterdir()):
        digit_source_folder = expected_upright_image_folder
    else:
        base_upright = Path(args.upright_out)
        found_match = False
        for c in crops:
            crop_name = Path(c).name
            if (base_upright / crop_name).exists():
                found_match = True
                break
        if found_match:
            digit_source_folder = base_upright
        else:
            digit_source_folder = crop_folder

    Path(args.digits_out).mkdir(parents=True, exist_ok=True)
    print(f"[main] Running digit inference on: {digit_source_folder}")
    call_digit_infer(args.digit_script, args.digit_model, digit_source_folder, args.digits_out, conf=args.conf)

    # Merge per-crop outputs into single file — ONLY for this image's stem
    final_digits = Path(args.digits_out) / "all_digits.txt"
    image_stem = Path(args.image).stem
    merge_digit_outputs(args.digits_out, final_digits, image_stem=image_stem)

    print("\n[main] 🎉 Pipeline complete!")
    print(" Crops saved to:     ", Path(args.out) / Path(args.image).stem)
    print(" Upright saved to:   ", args.upright_out)
    print(" Digit results saved:", args.digits_out)
    print(" Merged digits file: ", final_digits)

if __name__ == "__main__":
    main()
