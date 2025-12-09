#!/usr/bin/env python3
"""
detect_crop_upright_read_only.py

Single-image / video / folder pipeline:
 - Detect tags and crop (per-image or per-frame for video)
 - Run orient_and_fix.py on the collected crops -> outputs/upright
 - Run trainWithSizing.py --mode infer on upright -> outputs/digits

Usage examples:
# image:
python scripts/detect_crop_upright_read_only.py --model "models/detect/best.pt" --image "inputs/images/foo.jpg" --out "outputs/crops" --upright_out "outputs/upright" --digits_out "outputs/digits" --conf 0.1

# video:
python scripts/detect_crop_upright_read_only.py --model "models/detect/best.pt" --image "inputs/videos/video.mp4" --frame_step 5 --out "outputs/crops" --upright_out "outputs/upright" --digits_out "outputs/digits" --conf 0.1

# folder:
python scripts/detect_crop_upright_read_only.py --model "models/detect/best.pt" --image "inputs/images_folder" --out "outputs/crops" ...
"""

import argparse
import cv2
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_frames_from_video(video_path: Path, out_dir: Path, frame_step=1, max_frames=None):
    """Extract frames from a video into out_dir. Return number of frames saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[video] Failed to open video: {video_path}")
        return 0
    count = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[video] Extracting frames from {video_path.name} (approx {total}) -> {out_dir}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % frame_step != 0:
            continue
        fname = out_dir / f"{video_path.stem}_frame_{count:06d}.jpg"
        cv2.imwrite(str(fname), frame)
        saved += 1
        if max_frames and saved >= max_frames:
            break
    cap.release()
    print(f"[video] Saved {saved} frames from {video_path.name}")
    return saved


def crop_from_image(model_path, image_path, output_folder, conf=0.2, save_annotated=False):
    """Run YOLO on a single image and save crops. Returns list of saved crop paths."""
    model = YOLO(str(model_path))

    image_path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if not len(result.boxes):
        return []

    saved_paths = []
    for i, box in enumerate(result.boxes):
        # safe extract coordinates
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
        base = image_path.stem
        crop_fname = out_dir / f"{base}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_fname), crop)
        saved_paths.append(str(crop_fname))
    return saved_paths


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
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError("Orientation script failed.")
    print("[orient] Completed.")


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
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError("Digit inference script failed.")
    print("[digits] Completed.")


def main():
    p = argparse.ArgumentParser(description="Detect -> crop -> upright -> infer digits (accepts image, video, or folder)")
    p.add_argument("--model", required=True, help="path to tag detector weights")
    p.add_argument("--image", required=True, help="image file, video file, or folder of images")
    p.add_argument("--out", default="outputs/crops", help="where to save crops")
    p.add_argument("--upright_out", default="outputs/upright", help="where to save upright images")
    p.add_argument("--digits_out", default="outputs/digits", help="where to save digit json/txt")
    p.add_argument("--upright_model", default="models/upright/best.pt", help="orientation model weights")
    p.add_argument("--digit_model", default="models/read/best.pt", help="digit model weights")
    p.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="path to orient_and_fix.py")
    p.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="path to trainWithSizing.py")
    p.add_argument("--conf", type=float, default=0.1, help="detection confidence")
    p.add_argument("--device", default="cpu", help="device for orient script")
    p.add_argument("--frame_step", type=int, default=1, help="for video: keep every Nth frame")
    p.add_argument("--max_frames", type=int, default=None, help="for video: stop after extracting this many frames")
    p.add_argument("--keep_frames", action="store_true", help="keep extracted frames (default: delete)")
    args = p.parse_args()

    image_arg = Path(args.image)
    crops_folder = Path(args.out)
    upright_folder = Path(args.upright_out)
    digits_folder = Path(args.digits_out)

    crops_folder.mkdir(parents=True, exist_ok=True)
    upright_folder.mkdir(parents=True, exist_ok=True)
    digits_folder.mkdir(parents=True, exist_ok=True)

    temp_frames_dir = None
    process_paths = []

    # Determine input type
    if image_arg.is_dir():
        # folder of images
        img_list = sorted([p for p in image_arg.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        if not img_list:
            print("[main] No images found in folder:", image_arg)
            return
        process_paths = [str(p) for p in img_list]
    else:
        ext = image_arg.suffix.lower()
        if ext in VIDEO_EXTS:
            # extract frames
            if args.keep_frames:
                temp_frames_dir = Path(args.out).parent / "temp_frames"
                temp_frames_dir.mkdir(parents=True, exist_ok=True)
            else:
                tmp = tempfile.mkdtemp(prefix="dcuf_frames_")
                temp_frames_dir = Path(tmp)
            n = extract_frames_from_video(image_arg, temp_frames_dir, frame_step=args.frame_step, max_frames=args.max_frames)
            if n == 0:
                print("[main] No frames extracted from video. Exiting.")
                if not args.keep_frames and temp_frames_dir and temp_frames_dir.exists():
                    shutil.rmtree(temp_frames_dir)
                return
            # use frames
            process_paths = sorted([str(p) for p in temp_frames_dir.glob("*.jpg")])
        elif ext in IMAGE_EXTS:
            process_paths = [str(image_arg)]
        else:
            print("[main] Unsupported input type:", image_arg)
            return

    # Run detection & cropping on each path (image or extracted frame)
    total_crops = 0
    for pth in process_paths:
        try:
            crops = crop_from_image(args.model, pth, crops_folder, conf=args.conf)
            print(f"[main] {pth} -> {len(crops)} crops")
            total_crops += len(crops)
        except Exception as e:
            print(f"[main] Detection failed on {pth}: {e}")

    if total_crops == 0:
        print("[main] No crops produced from any inputs. Exiting.")
        if temp_frames_dir and temp_frames_dir.exists() and not args.keep_frames:
            shutil.rmtree(temp_frames_dir)
        return

    # Run orientation on the entire crops folder
    call_orient_script(args.orient_script, args.upright_model, crops_folder, upright_folder, device=args.device, accept_conf=args.conf)

    # Run digit inference on upright folder
    call_digit_infer(args.digit_script, args.digit_model, upright_folder, digits_folder, conf=args.conf)

    # cleanup frames if we created a temp folder and user didn't ask to keep frames
    if temp_frames_dir and temp_frames_dir.exists() and not args.keep_frames:
        try:
            shutil.rmtree(temp_frames_dir)
        except Exception:
            pass

    print("\n[main] Pipeline complete.")
    print("Crops ->", crops_folder)
    print("Upright ->", upright_folder)
    print("Digits ->", digits_folder)


if __name__ == "__main__":
    main()
