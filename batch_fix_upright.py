#!/usr/bin/env python3
"""
batch_fix_upright_ocr.py

Evaluate rotations (0,90,180,270) and optional flips for every image in a flat folder.
Choose the transform that maximizes numeric OCR confidence (prefers readable digits).
Fallback to gradient+aspect heuristic when OCR finds nothing.

Usage:
  python batch_fix_upright_ocr.py --src "<src_folder>" --out "<out_folder>" [--flips] [--verbose]

Notes:
 - Installs: pip install opencv-python pillow pytesseract easyocr
 - If both pytesseract and easyocr present, pytesseract is tried first (fast), easyocr used if pytesseract yields nothing.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import sys
from PIL import Image
import tempfile
import re

NUMERIC_RE = re.compile(r'[\d\-]+')
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Try imports
try:
    import pytesseract
    HAS_PYTESS = True
except Exception:
    pytesseract = None
    HAS_PYTESS = False

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    easyocr = None
    HAS_EASYOCR = False

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

def rotate_image_cv(img, angle):
    if angle == 0:
        return img.copy()
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def flip_image_cv(img, mode):
    if not mode:
        return img.copy()
    if mode == "h":
        return cv2.flip(img, 1)
    if mode == "v":
        return cv2.flip(img, 0)
    if mode == "hv":
        return cv2.flip(img, -1)
    return img.copy()

def enhance_readability(bgr):
    try:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y2 = clahe.apply(y)
        ycc2 = cv2.merge([y2, cr, cb])
        base = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR)
        blur = cv2.GaussianBlur(base, (0,0), 1.0)
        sharp = cv2.addWeighted(base, 1.5, blur, -0.5, 0)
        den = cv2.bilateralFilter(sharp, d=5, sigmaColor=20, sigmaSpace=20)
        return den
    except Exception:
        return bgr

def score_variant_gradient(img):
    h, w = img.shape[:2]
    if h==0 or w==0: return -1e9
    aspect = float(h)/float(w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mean_abs_gx = float(np.mean(np.abs(gx)))
    mean_abs_gy = float(np.mean(np.abs(gy)))
    grad_ratio = (mean_abs_gx + 1e-6) / (mean_abs_gy + 1e-6)
    return 3.0 * grad_ratio + 2.0 * aspect

def easyocr_from_bytes(buf, reader=None):
    if not HAS_EASYOCR:
        return []
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
            tf.write(buf)
            tf.flush()
            r = reader or easyocr.Reader(['en'], gpu=False)
            raw = r.readtext(tf.name, detail=1, paragraph=False)
            out = []
            for box,text,conf in raw:
                out.append((text.strip(), float(conf) if conf is not None else 0.0))
            return out
    except Exception:
        return []

def pytess_from_pil(pil_img):
    if not HAS_PYTESS:
        return []
    try:
        txt = pytesseract.image_to_string(pil_img, config="--psm 6")
        txt = txt.strip()
        return [(txt, 0.5)] if txt else []
    except Exception:
        return []

def ocr_score(results):
    if not results:
        return 0.0
    s = 0.0
    for t,c in results:
        if NUMERIC_RE.search(t):
            s += 2.0 * max(0.0, float(c))
        else:
            s += 0.4 * max(0.0, float(c))
    s += 0.05 * len(results)
    return float(s)

def run_ocr_on_candidate(img_bgr, easy_reader=None):
    """
    Returns (ocr_results_list, ocr_score, mean_conf)
    """
    res = []
    mean_conf = 0.0
    # try pytesseract first (fast) using PIL
    if HAS_PYTESS:
        try:
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            res = pytess_from_pil(pil)
        except Exception:
            res = []
    # if pytesseract found nothing and easyocr exists, try easyocr
    if (not res) and HAS_EASYOCR:
        try:
            _, buf = cv2.imencode('.png', img_bgr)
            res = easyocr_from_bytes(buf.tobytes(), reader=easy_reader)
        except Exception:
            res = res or []
    if res:
        mean_conf = sum([c for _,c in res]) / max(1, len(res))
        return res, ocr_score(res), mean_conf
    return [], 0.0, 0.0

def pick_best_transform_for_image(img_bgr, allow_flips=False, easy_reader=None):
    rotations = [0,90,180,270]
    flips = [None] if not allow_flips else [None, "h", "v", "hv"]
    best = {"primary": -1e9, "secondary": -1e9, "angle":0, "flip":None, "img":img_bgr}
    # primary = OCR numeric score, secondary = gradient fallback score
    for angle in rotations:
        r = rotate_image_cv(img_bgr, angle)
        for f in flips:
            cand = flip_image_cv(r, f)
            # get OCR-based score
            ocr_res, ocr_s, mean_conf = run_ocr_on_candidate(cand, easy_reader=easy_reader)
            if ocr_s > 0.0:
                # prefer higher OCR numeric score; small penalty for flips
                flip_penalty = 0.2 if f is None else 0.6
                score = ocr_s - flip_penalty
                # tie-breaker: mean_conf
                if (score > best["primary"]) or (score==best["primary"] and mean_conf > best["secondary"]):
                    best.update({"primary": score, "secondary": mean_conf, "angle":angle, "flip":f, "img":cand.copy()})
            else:
                # no OCR: use gradient/aspect as fallback (lower priority)
                g = score_variant_gradient(cand)
                # keep best fallback if no OCR candidate exists yet
                if best["primary"] <= 0.0:  # only compare gradient when no OCR winner so far
                    if g > best["secondary"]:
                        best.update({"primary": 0.0, "secondary": g, "angle":angle, "flip":f, "img":cand.copy()})
    return best

def score_variant_gradient(img):
    # same as earlier gradient scoring function
    h, w = img.shape[:2]
    if h==0 or w==0: return -1e9
    aspect = float(h)/float(w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mean_abs_gx = float(np.mean(np.abs(gx)))
    mean_abs_gy = float(np.mean(np.abs(gy)))
    grad_ratio = (mean_abs_gx + 1e-6) / (mean_abs_gy + 1e-6)
    return 3.0 * grad_ratio + 2.0 * aspect

def process_all(src_path, out_path, allow_flips=False, verbose=False):
    src = Path(src_path)
    out = Path(out_path)
    if not src.exists() or not src.is_dir():
        print("Source folder missing:", src); return
    out.mkdir(parents=True, exist_ok=True)
    files = list_images(src)
    if not files:
        print("No images found in", src); return
    easy_reader = easyocr.Reader(['en'], gpu=False) if (HAS_EASYOCR) else None
    total = len(files)
    for i,p in enumerate(files, start=1):
        try:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[{i}/{total}] failed read: {p.name}"); continue
            best = pick_best_transform_for_image(img, allow_flips=allow_flips, easy_reader=easy_reader)
            out_img = out / f"upright_{p.stem}.jpg"
            out_enh = out / f"upright_{p.stem}_enh.jpg"
            cv2.imwrite(str(out_img), best["img"])
            cv2.imwrite(str(out_enh), enhance_readability(best["img"]))
            if verbose:
                print(f"[{i}/{total}] {p.name} -> angle={best['angle']} flip={best['flip']} primary_score={best['primary']:.3f} secondary={best['secondary']:.3f}")
        except Exception as e:
            print(f"[{i}/{total}] error {p.name}: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flips", action="store_true", help="consider mirrored flips")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not HAS_PYTESS and not HAS_EASYOCR:
        print("Warning: neither pytesseract nor easyocr found; script will use gradient fallback only.")
    process_all(args.src, args.out, allow_flips=args.flips, verbose=args.verbose)
