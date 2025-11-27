#!/usr/bin/env python3
r"""
generate_rotations_and_flips.py

Generate rotated (0,90,180,270) and flipped variants of all images in a folder.

Usage example (PowerShell):
  python generate_rotations_and_flips.py `
    --src "C:\Users\Gnanasekar\Downloads\MODELS\My First Project.v5i.yolov8 - ultralytics-colab\dataset\images\train\format part tag cropped" `
    --out "C:\path\to\out_folder" --flips

Notes:
- If --flips is provided, the script will also produce horizontal, vertical and both flips
  for every rotation (this produces up to 16 files per input image).
- Filenames are produced as: {stem}_r{angle}[_flip_h][_flip_v].{ext}
- The script will create the output folder if it doesn't exist.
"""

import argparse
from pathlib import Path
import cv2
import os
import sys

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def rotate_image_cv(img, angle):
    """Rotate OpenCV BGR image by angle (0,90,180,270)."""
    if angle == 0:
        return img.copy()
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # fallback (shouldn't happen)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def flip_image_cv(img, mode):
    """Flip image: mode in {None, 'h', 'v', 'hv'}"""
    if not mode:
        return img.copy()
    if mode == "h":
        return cv2.flip(img, 1)
    if mode == "v":
        return cv2.flip(img, 0)
    if mode == "hv":
        return cv2.flip(img, -1)
    return img.copy()


def iter_images(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def make_output_name(src_path: Path, angle: int, flip_mode: str):
    stem = src_path.stem
    ext = src_path.suffix.lower()
    parts = [f"{stem}_r{angle}"]
    if flip_mode == "h":
        parts.append("flip_h")
    elif flip_mode == "v":
        parts.append("flip_v")
    elif flip_mode == "hv":
        parts.append("flip_hv")
    name = "_".join(parts) + ext
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source folder containing images.")
    p.add_argument("--out", required=True, help="Output folder to write variants.")
    p.add_argument("--flips", action="store_true", help="Also produce flipped variants (h, v, hv).")
    p.add_argument("--skip-existing", action="store_true", help="Skip writing files that already exist in output.")
    p.add_argument("--verbose", action="store_true", help="Print each file written.")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists() or not src.is_dir():
        print("Source folder not found or not a directory:", src)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    rotations = [0, 90, 180, 270]
    flips = [None]
    if args.flips:
        flips = [None, "h", "v", "hv"]

    total_in = 0
    total_written = 0

    for img_path in iter_images(src):
        total_in += 1
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                if args.verbose:
                    print("Failed to read (skipping):", img_path)
                continue
        except Exception as e:
            if args.verbose:
                print("Error reading", img_path, ":", e)
            continue

        for angle in rotations:
            rotated = rotate_image_cv(img, angle)
            for flip_mode in flips:
                transformed = flip_image_cv(rotated, flip_mode)
                out_name = make_output_name(img_path, angle, flip_mode)
                out_path = out / out_name
                if args.skip_existing and out_path.exists():
                    if args.verbose:
                        print("Skipping existing:", out_path.name)
                    continue
                # write with reasonable JPEG quality for jpg
                try:
                    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                        cv2.imwrite(str(out_path), transformed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    else:
                        cv2.imwrite(str(out_path), transformed)
                    total_written += 1
                    if args.verbose:
                        print("Wrote:", out_path.name)
                except Exception as e:
                    print("Failed to write", out_path, ":", e)

    print(f"\nDone. Input images: {total_in}. Variants written: {total_written}. Output folder: {out}")


if __name__ == "__main__":
    main()
