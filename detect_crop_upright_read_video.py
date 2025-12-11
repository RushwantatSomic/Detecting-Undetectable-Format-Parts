#!/usr/bin/env python3
"""
detect_crop_upright_digits_video.py

Video pipeline:
1. Extract frames (configurable step)
2. Detect tags on each sampled frame -> save crops to per-video folder (video_stem + frame index in filenames)
3. Orient/upright those crops (point orient script to the per-video crop folder)
4. Run digit inference on oriented outputs -> save digit outputs in per-video digits folder
5. Merge per-video digit outputs into per-video all_digits.txt (deduplicated, sorted)

Usage example:
python scripts/detect_crop_upright_digits_video.py \
  --model models/detect/best.pt \
  --video inputs/videos/test.mp4 \
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
import re
from collections import OrderedDict

# -----------------------
# Merge helper (dedupe + sort by _crop_XXXX)
# -----------------------
def merge_digit_outputs(digits_root_path, merged_output_path, name_prefix=None):
    """
    Merge per-crop digit outputs (deduplicated).

    digits_root_path: folder to recursively search for .txt/.json files
    merged_output_path: path (file) to write final lines: "<tag_filename> : <digits...>"
    name_prefix: optional string; if provided only merge entries whose image/crop filename contains this prefix.
                 For video runs, this should be the video_stem (e.g. 'myvideo').
    """
    digits_root = Path(digits_root_path)
    merged_path = Path(merged_output_path)

    entries = OrderedDict()  # img_name -> digits_str (keeps last-seen by default)

    def store(img_name, digits_str):
        if name_prefix and name_prefix not in img_name:
            return
        img_name = img_name.strip()
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
                m = re.search(r"image[:]\s*(\S+)", content, flags=re.I)
                if m:
                    candidate_name = m.group(1).strip()
                store(candidate_name, content)
        except Exception:
            continue

    # Fallback to JSON files if no txt entries or additional entries
    json_files = sorted(digits_root.rglob("*.json"))
    for j in json_files:
        try:
            raw = j.read_text(encoding="utf-8")
            data = json.loads(raw)
            img_name = data.get("image") or data.get("filename") or j.stem
            dets = data.get("detections", [])
            dets_sorted = sorted(dets, key=lambda d: (d.get("bbox")[0] if d.get("bbox") else 0))
            digits = " ".join(str(int(d.get("class"))) for d in dets_sorted if "class" in d)
            store(str(img_name), digits)
        except Exception:
            continue

    if not entries:
        print(f"[merge] No digit outputs found under {digits_root} for prefix={name_prefix}")
        return None

    # Sort by crop index if names contain '_crop_XXXX' pattern
    def crop_index_key(name):
        m = re.search(r"_crop_(\d{1,6})", name)
        if m:
            return int(m.group(1))
        # fallback keep insertion order for items without index
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
# STEP A — EXTRACT & SAMPLE FRAMES
# ----------------------------------------------------------
def extract_sampled_frames(video_path, frames_out_folder, frame_step=1, max_frames=None):
    """
    Extract frames from video_path and save sampled frames into frames_out_folder.
    Returns list of saved frame file paths (Path objects).

    frame_step: sample every N frames (1 = every frame).
    max_frames: optional int to limit number of extracted frames (for debugging).
    """
    video_path = Path(video_path)
    frames_out_folder = Path(frames_out_folder)
    frames_out_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    saved = []
    frame_idx = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[frames] Video {video_path.name} opened, total_frames={total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            fname = frames_out_folder / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved.append(fname)
            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"[frames] Extracted {len(saved)} frames into {frames_out_folder}")
    return saved

# ----------------------------------------------------------
# STEP B — DETECT & CROP FOR VIDEO (all crops go into video crop folder)
# ----------------------------------------------------------
def crop_from_frames(model_path, frames_list, output_folder, conf=0.2):
    """
    Runs tag detection on each frame and writes crops into:
      <output_folder>/<video_stem>_frame_{frameidx}_crop_{i:04d}.jpg
    frames_list: iterable of Path to frame images
    Returns list of created crop file paths (strings).
    """
    print(f"[detect-video] Loading detector: {model_path}")
    model = YOLO(str(model_path))

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for frame_path in frames_list:
        frame_path = Path(frame_path)
        results = model.predict(str(frame_path), conf=conf, verbose=False)
        result = results[0]
        if not getattr(result, "boxes", None) or len(result.boxes) == 0:
            continue

        # read img to crop using precise coords (YOLO boxes can be used)
        import numpy as np
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

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

            crop_filename = out_dir / f"{frame_path.stem}_crop_{i+1:04d}.jpg"
            cv2.imwrite(str(crop_filename), crop)
            saved_paths.append(str(crop_filename))
        print(f"[detect-video] Saved {len(result.boxes)} crops from {frame_path.name}")

    print(f"[detect-video] Cropping complete. Total crops saved: {len(saved_paths)}")
    return saved_paths

# ----------------------------------------------------------
# STEP C — ORIENT & DIGIT (wrapper callers)
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
# MAIN VIDEO PIPELINE
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video: detect tags → crop → upright → digit detect (per-video outputs)")
    parser.add_argument("--model", required=True, help="Path to TAG detector model")
    parser.add_argument("--video", required=True, help="Video to process (mp4, avi, etc.)")

    parser.add_argument("--out_video_crops", default="outputs/video_crops", help="Base folder to save video crops (per-video subfolder created)")
    parser.add_argument("--upright_video_out", default="outputs/upright_video", help="Base folder for upright outputs for videos")
    parser.add_argument("--digits_video_out", default="outputs/digits_video", help="Base folder for digit outputs for videos")

    parser.add_argument("--upright_model", default="models/upright/best.pt", help="Upright model weights")
    parser.add_argument("--digit_model", default="models/read/best.pt", help="Digit detection model weights")

    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="Path to orient script")
    parser.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="Path to digit model script")

    parser.add_argument("--conf", type=float, default=0.1, help="Confidence for detection")
    parser.add_argument("--device", default="cpu", help="Device for upright model")

    parser.add_argument("--frame_step", type=int, default=1, help="Sample every N frames from the video (1 = every frame)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional limit to number of frames to extract (useful for debugging)")

    args = parser.parse_args()

    video_path = Path(args.video)
    video_stem = video_path.stem

    # Frames folder for this video (temporary frames, saved separately)
    frames_folder = Path("tmp_video_frames") / video_stem
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Step A: extract frames
    frames = extract_sampled_frames(video_path, frames_folder, frame_step=args.frame_step, max_frames=args.max_frames)
    if not frames:
        print("[main-video] No frames extracted. Stopping.")
        return

    # Step B: detect & crop (all crops for this video into a per-video folder)
    video_crop_folder = Path(args.out_video_crops) / video_stem
    video_crop_folder.mkdir(parents=True, exist_ok=True)
    crops = crop_from_frames(args.model, frames, video_crop_folder, conf=args.conf)
    if not crops:
        print("[main-video] No crops produced from video. Stopping.")
        return

    # Step C: orient (point to the per-video crop folder)
    call_orient_script(args.orient_script, args.upright_model, video_crop_folder, args.upright_video_out, device=args.device, accept_conf=args.conf)

    # After orient, expect upright images to be under upright_video_out/<video_stem>/ (many orient scripts mirror input name)
    expected_upright_folder = Path(args.upright_video_out) / video_stem
    base_upright = Path(args.upright_video_out)
    if expected_upright_folder.exists() and any(expected_upright_folder.iterdir()):
        digit_source_folder = expected_upright_folder
    else:
        # check if upright outputs have files matching our crop names directly under base
        found_match = False
        for c in crops[:10]:  # just sample a few for quick check
            if (base_upright / Path(c).name).exists():
                found_match = True
                break
        digit_source_folder = base_upright if found_match else video_crop_folder

    # Step D: digit inference (per-video digits output folder)
    video_digits_folder = Path(args.digits_video_out) / video_stem
    video_digits_folder.mkdir(parents=True, exist_ok=True)
    print(f"[main-video] Running digit inference on: {digit_source_folder}")
    call_digit_infer(args.digit_script, args.digit_model, digit_source_folder, video_digits_folder, conf=args.conf)

    # Step E: merge per-video outputs only (use name_prefix=video_stem)
    final_digits = video_digits_folder / "all_digits.txt"
    merge_digit_outputs(video_digits_folder, final_digits, name_prefix=video_stem)

    print("\n[main-video] 🎉 Video pipeline complete!")
    print(" Video crops saved to: ", video_crop_folder)
    print(" Upright base folder:   ", args.upright_video_out)
    print(" Digit results saved:   ", video_digits_folder)
    print(" Merged digits file:    ", final_digits)
    print(" Temp frames folder:    ", frames_folder)

if __name__ == "__main__":
    main()
