#!/usr/bin/env python3
"""
detect_crop_upright_read_images.py

Process ALL images inside a folder. For each image:
 1) detect tags -> save crops in out/crops/<image_stem>/
 2) run orient_and_fix.py -> save upright in out/upright/<image_stem>/
 3) run trainWithSizing.py --mode infer -> save results in out/digits/<image_stem>/

Example:
python scripts/detect_crop_upright_read_images.py --images_folder inputs/images --out outputs \
  --model models/detect/best.pt --upright_model models/upright/best.pt --digit_model models/read/best.pt --conf 0.1
"""
import argparse
import subprocess
import sys
from pathlib import Path
import traceback
import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def crop_from_image(model_path, image_path, output_folder, conf=0.2):
    """Run YOLO on a single image and save crops. Returns list of crop paths."""
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

        crop_fname = out_dir / f"{image_path.stem}_crop_{i+1:04d}.jpg"
        cv2.imwrite(str(crop_fname), crop)
        saved_paths.append(str(crop_fname))

    return saved_paths


def run_subprocess(cmd):
    """Run command, print output, return (returncode, output_text)."""
    print(">>>", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode, proc.stdout


def call_orient_script(orient_script_path, weights_path, source_folder, output_folder, device="cpu", accept_conf=0.35):
    orient_script = Path(orient_script_path)
    if not orient_script.exists():
        raise FileNotFoundError(f"orient script not found: {orient_script}")
    cmd = [
        sys.executable,
        str(orient_script),
        "--weights", str(weights_path),
        "--source", str(source_folder),
        "--output", str(output_folder),
        "--device", str(device),
        "--accept-conf", str(accept_conf)
    ]
    rc, out = run_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(f"orient script failed (rc={rc})")


def call_digit_infer(digit_script_path, weights_path, source_folder, output_folder, conf=0.35):
    digit_script = Path(digit_script_path)
    if not digit_script.exists():
        raise FileNotFoundError(f"digit inference script not found: {digit_script}")
    cmd = [
        sys.executable,
        str(digit_script_path),
        "--mode", "infer",
        "--weights", str(weights_path),
        "--input_folder", str(source_folder),
        "--out_folder", str(output_folder),
        "--conf", str(conf)
    ]
    rc, out = run_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(f"digit inference failed (rc={rc})")


def process_one_image(image_path, args):
    stem = Path(image_path).stem
    crops_dir = Path(args.out) / "crops" / stem
    upright_dir = Path(args.out) / "upright" / stem
    digits_dir = Path(args.out) / "digits" / stem

    crops_dir.mkdir(parents=True, exist_ok=True)
    upright_dir.mkdir(parents=True, exist_ok=True)
    digits_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing image: {image_path} ===")
    # 1) detect & crop
    try:
        crops = crop_from_image(args.model, image_path, crops_dir, conf=args.conf)
        print(f"-> crops found: {len(crops)} (saved to {crops_dir})")
        if not crops:
            print(f"[WARN] No crops for {image_path}. Skipping orient & digit steps.")
            return {"image": str(image_path), "crops": 0, "upright": 0, "digits": 0, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] detection failed for {image_path}:\n{tb}")
        return {"image": str(image_path), "crops": 0, "upright": 0, "digits": 0, "error": str(e)}

    # 2) orient (on that image's crop folder)
    try:
        call_orient_script(args.orient_script, args.upright_model, crops_dir, upright_dir, device=args.device, accept_conf=args.conf)
        upright_count = len(list(upright_dir.glob("*.*")))
        print(f"-> upright saved: {upright_count} files in {upright_dir}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] orient failed for {image_path}:\n{tb}")
        return {"image": str(image_path), "crops": len(crops), "upright": 0, "digits": 0, "error": str(e)}

    # 3) digit inference
    try:
        call_digit_infer(args.digit_script, args.digit_model, upright_dir, digits_dir, conf=args.conf)
        digit_files = len(list(digits_dir.glob("*")))
        print(f"-> digit outputs: {digit_files} files in {digits_dir}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] digit inference failed for {image_path}:\n{tb}")
        return {"image": str(image_path), "crops": len(crops), "upright": upright_count, "digits": 0, "error": str(e)}

    return {"image": str(image_path), "crops": len(crops), "upright": upright_count, "digits": digit_files, "error": None}


def main():
    parser = argparse.ArgumentParser(description="Batch: detect -> crop -> upright -> digits for all images in a folder")
    parser.add_argument("--images_folder", required=True, help="folder containing images (recursively searched)")
    parser.add_argument("--out", default="outputs", help="root output folder (creates crops/, upright/, digits/ subfolders grouped by image)")
    parser.add_argument("--model", default="models/detect/best.pt", help="tag detector weights")
    parser.add_argument("--upright_model", default="models/upright/best.pt", help="orientation model weights")
    parser.add_argument("--digit_model", default="models/read/best.pt", help="digit model weights")
    parser.add_argument("--orient_script", default="scripts/orient_and_fix.py", help="path to orient_and_fix.py")
    parser.add_argument("--digit_script", default="scripts/trainWithSizing.py", help="path to trainWithSizing.py")
    parser.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    parser.add_argument("--device", default="cpu", help="device string for orient script")
    parser.add_argument("--max_images", type=int, default=None, help="limit number of images processed (for testing)")
    args = parser.parse_args()

    images_folder = Path(args.images_folder)
    if not images_folder.exists():
        raise SystemExit(f"Images folder not found: {images_folder}")

    # collect images recursively
    image_list = sorted([p for p in images_folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not image_list:
        raise SystemExit(f"No images found in {images_folder}")

    # optional limit (for fast test)
    if args.max_images:
        image_list = image_list[: args.max_images]

    print(f"Found {len(image_list)} images. Starting batch pipeline...")

    summary = []
    for img_path in image_list:
        res = process_one_image(str(img_path), args)
        summary.append(res)

    # print small summary
    print("\nBatch summary:")
    for s in summary:
        print(f"- {Path(s['image']).name}: crops={s['crops']} upright={s['upright']} digits={s['digits']} error={s['error']}")

    print("\nAll done. Outputs are under:", Path(args.out).resolve())


if __name__ == "__main__":
    import argparse
    main()
