#!/usr/bin/env python3
r"""
pick_upright_from_debug.py

Works with:
 - debug root containing per-sample subfolders (old behavior), OR
 - debug root that is a flat folder with images (NEW: each image becomes a sample)

Usage:
  python pick_upright_from_debug.py --debug "autoorient_debug" --out "debug_final" [--no-ocr] [--no-flips] [--force-all]

Output:
  out_dir/
    upright_<stem>.jpg
    upright_<stem>_enh.jpg
    <stem>_candidates.json
  upright_final_log.csv
"""
import argparse
import csv
import json
import sys
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# OCR backends detection
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    easyocr = None
    _HAS_EASYOCR = False

try:
    import pytesseract
    _HAS_PYTESS = True
except Exception:
    pytesseract = None
    _HAS_PYTESS = False

NUMERIC_FIND_RE = re.compile(r"[\d\-]+")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"):
        imgs += sorted(folder.glob(ext))
    return imgs


def read_img_cv(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return img


# OCR helpers
def easyocr_read_from_bytes(buf: bytes, reader=None):
    if not _HAS_EASYOCR:
        return []
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
            tf.write(buf)
            tf.flush()
            reader = reader or easyocr.Reader(["en"], gpu=False)
            raw = reader.readtext(tf.name, detail=1, paragraph=False)
    except Exception:
        return []
    out = []
    for box, text, conf in raw:
        try:
            out.append((text.strip(), float(conf)))
        except Exception:
            out.append((text.strip(), 0.0))
    return out


def tesseract_read_from_pil(pil_img: Image.Image):
    if not _HAS_PYTESS:
        return []
    try:
        txt = pytesseract.image_to_string(pil_img, config="--psm 6")
        txt = txt.strip()
        return [(txt, 0.5)] if txt else []
    except Exception:
        return []


def ocr_score(results: List[Tuple[str, float]]) -> float:
    if not results:
        return 0.0
    s = 0.0
    for t, c in results:
        if NUMERIC_FIND_RE.search(t):
            s += 2.0 * max(0.0, float(c))
        else:
            s += 0.4 * max(0.0, float(c))
    s += 0.05 * len(results)
    return float(s)


def extract_best_numeric_from_ocr(res: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not res:
        return ("", 0.0)
    candidates = []
    for t, c in res:
        m = NUMERIC_FIND_RE.findall(t)
        if m:
            candidates.append((max(m, key=len), c))
    if candidates:
        candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        return candidates[0]
    res.sort(key=lambda x: x[1], reverse=True)
    return (res[0][0], res[0][1])


def enhance_readability(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y)
    ycc2 = cv2.merge([y2, cr, cb])
    base = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    sharp = cv2.addWeighted(base, 1.5, blur, -0.5, 0)
    den = cv2.bilateralFilter(sharp, d=5, sigmaColor=20, sigmaSpace=20)
    return den


# transforms
def rotate_image_cv(img, angle: int):
    if angle == 0:
        return img.copy()
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def flip_image_cv(img, mode: Optional[str]):
    if not mode:
        return img.copy()
    if mode == "h":
        return cv2.flip(img, 1)
    if mode == "v":
        return cv2.flip(img, 0)
    if mode == "hv":
        return cv2.flip(img, -1)
    return img.copy()


# improved evaluator (gradient + aspect + OCR, flip penalty)
def evaluate_image_variants(img_path: Path, use_ocr: bool, ocr_reader=None, allow_flips: bool = True):
    """
    Evaluate rotated and flipped variants of the image.
    Returns dict with keys: score, angle, flip, best_img (BGR numpy), ocr (list).
    """
    try:
        img_orig = read_img_cv(img_path)
    except Exception:
        return None

    rotations = [0, 90, 180, 270]
    flips = [None] if not allow_flips else [None, "h", "v", "hv"]

    # baseline OCR on original (cheap) to compare flip improvements
    baseline_ocr_conf = 0.0
    baseline_count = 0
    if use_ocr and _HAS_PYTESS:
        try:
            pil_orig = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            r = tesseract_read_from_pil(pil_orig)
            if r:
                baseline_count = len(r)
                baseline_ocr_conf = sum([c for _, c in r]) / len(r)
        except Exception:
            baseline_ocr_conf = 0.0

    best = {"score": -1e9, "angle": 0, "flip": None, "best_img": None, "ocr": []}

    for angle in rotations:
        rotated = rotate_image_cv(img_orig, angle)
        # gradient measures
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mean_abs_gx = float(np.mean(np.abs(gx)))
        mean_abs_gy = float(np.mean(np.abs(gy)))
        grad_ratio = (mean_abs_gx + 1e-6) / (mean_abs_gy + 1e-6)

        for flip in flips:
            cand = flip_image_cv(rotated, flip)
            h, w = cand.shape[:2]
            if w == 0:
                continue
            aspect = float(h) / float(w)

            # orientation mix: aspect and gradient ratio (gradient dominates)
            orient_score = 2.0 * aspect + 3.0 * grad_ratio

            # OCR: pytesseract first (cheap), then easyocr if available and needed
            ocr_res = []
            ocr_conf_mean = 0.0

            if use_ocr and _HAS_PYTESS:
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
                    ocr_res = tesseract_read_from_pil(pil_img)
                    if ocr_res:
                        ocr_conf_mean = sum([c for _, c in ocr_res]) / len(ocr_res)
                except Exception:
                    ocr_res = []

            if use_ocr and _HAS_EASYOCR:
                try:
                    _, buf = cv2.imencode('.png', cand)
                    easy_res = easyocr_read_from_bytes(buf.tobytes(), reader=ocr_reader)
                    if easy_res:
                        ocr_res = easy_res
                        ocr_conf_mean = sum([c for _, c in ocr_res]) / len(ocr_res) if ocr_res else 0.0
                except Exception:
                    pass

            ocr_s = ocr_score(ocr_res) if ocr_res else 0.0

            # combine scores (orientation dominates)
            total_score = orient_score * 2.5 + ocr_s

            # strong flip penalty unless flip significantly increases OCR mean conf vs baseline
            flip_penalty = 0.7
            if flip is not None:
                if use_ocr and baseline_ocr_conf > 0:
                    if (ocr_conf_mean - baseline_ocr_conf) > 0.20:
                        flip_adjust = -0.15  # reward flip a bit if big OCR gain
                    else:
                        flip_adjust = flip_penalty
                else:
                    flip_adjust = flip_penalty
                total_score -= flip_adjust

            if flip == "hv":
                total_score -= 0.2

            # update best
            if total_score > best["score"]:
                best.update({
                    "score": float(total_score),
                    "angle": int(angle),
                    "flip": flip,
                    "best_img": cand.copy(),
                    "ocr": ocr_res
                })

    return best


# wrapper for backward compatibility (older callers expect this signature)
def score_candidate_image(img_path: Path, use_ocr: bool, ocr_reader=None):
    info = evaluate_image_variants(img_path, use_ocr, ocr_reader)
    if not info:
        return 0.0, []
    return float(info["score"]), info["ocr"]


# picker -- now accepts flat or subfolder mode
def pick_best_candidates_from_folder(subfolder: Path, use_ocr: bool, ocr_reader=None, force_all=False, min_score=0.01, allow_flips=True):
    stem = subfolder.name
    files = list_images(subfolder)
    if not files:
        return {"stem": stem, "candidates": [], "chosen": None, "reason": "no_images"}

    # prefer rotation-only candidates (old behavior); but in flat mode this may not apply
    rot_re = re.compile(r"(?:_|-|\b)r(?:0|90|180|270)(?:_|\.|$)", flags=re.IGNORECASE)
    rot_matches = [p for p in files if rot_re.search(p.name)]

    if rot_matches and not force_all:
        used_list = sorted(rot_matches)
        used_reason = "rotation_only"
    else:
        good = []
        stem_low = stem.lower()
        for p in files:
            name_low = p.name.lower()
            if stem_low in name_low or name_low.startswith(stem_low + "_"):
                good.append(p)
        if good and not force_all:
            used_list = good
            used_reason = "strict_match"
        else:
            used_list = files
            used_reason = "forced_all" if force_all else "fallback_all"

    candidates_info = []
    best_variant = None
    best_score = -1e9
    best_orig_p = None

    for p in used_list:
        variant = evaluate_image_variants(p, use_ocr, ocr_reader, allow_flips=allow_flips)
        sc = variant["score"] if variant else 0.0
        ocr_list = variant.get("ocr", []) if variant else []
        candidates_info.append({
            "file": str(p.name),
            "score": float(sc),
            "ocr_count": len(ocr_list),
            "ocr_sample": ocr_list[:3],
            "chosen_variant": {"angle": variant.get("angle"), "flip": variant.get("flip")} if variant else {}
        })
        if sc > best_score:
            best_score = sc
            best_variant = variant
            best_orig_p = p

    chosen_info = None
    chosen_extra = None
    if best_variant and best_orig_p:
        chosen_info = {"file": str(best_orig_p.name), "score": float(best_score)}
        chosen_extra = {"angle": best_variant["angle"], "flip": best_variant["flip"], "img_bgr": best_variant["best_img"], "ocr": best_variant["ocr"]}

    return {
        "stem": stem,
        "reason": used_reason,
        "candidates": candidates_info,
        "chosen": chosen_info,
        "chosen_extra": chosen_extra,
        "chosen_ocr": chosen_extra["ocr"] if chosen_extra else []
    }


# process_debug_root: supports flat or subfolder layout
def process_debug_root(debug_root: Path, out_dir: Path, force_all=False, use_ocr=True, allow_flips=True, skip_threshold=0.6, min_score=0.01, csv_out: Path = None):
    ensure_dir(out_dir)

    ocr_reader = None
    if use_ocr and _HAS_EASYOCR:
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            ocr_reader = None

    # detect mode:
    subs = [p for p in sorted(debug_root.iterdir()) if p.is_dir()]
    flat_mode = False
    if not subs:
        # flat folder of images -> each image is its own sample
        files = list_images(debug_root)
        if not files:
            print("No images in debug root:", debug_root)
            return
        flat_mode = True
        subs = files  # in flat_mode, subs holds file paths

    rows = []

    for sub in subs:
        if flat_mode:
            # sub is a Path to an image file
            stem = sub.stem
            print(f"\n[STEM] {stem}  (flat image mode)")
            # evaluate variants for this single file directly
            variant = evaluate_image_variants(sub, use_ocr, ocr_reader, allow_flips=allow_flips)
            if not variant:
                print("  Failed to evaluate image, skipping.")
                rows.append({"stem": stem, "chosen": "", "score": 0.0, "ocr_text": "", "ocr_conf": 0.0, "match_method": "eval_error"})
                continue

            # irrespective of identity, always write an upright image (transformed if available)
            try:
                transformed = variant.get("best_img") or read_img_cv(sub)
                out_img = out_dir / f"upright_{stem}.jpg"
                out_enh = out_dir / f"upright_{stem}_enh.jpg"
                cv2.imwrite(str(out_img), transformed)
                cv2.imwrite(str(out_enh), enhance_readability(transformed))

                ocr_text, ocr_conf = ("", 0.0)
                if variant.get("ocr"):
                    ocr_text, ocr_conf = extract_best_numeric_from_ocr(variant.get("ocr"))

                angle = variant.get("angle", 0)
                flipmode = variant.get("flip", None)
                score = float(variant.get("score", 0.0))

                print(f"  [SAVED] {sub.name} -> angle={angle} flip={flipmode} score={score:.3f} ocr_text='{ocr_text}'")
                rows.append({"stem": stem, "chosen": str(out_img.name), "score": score, "ocr_text": ocr_text, "ocr_conf": float(ocr_conf), "match_method": "flat_single"})
            except Exception as e:
                print("  [ERROR] saving transformed:", e)
                rows.append({"stem": stem, "chosen": "error", "score": float(variant.get("score", 0.0)), "ocr_text": "", "ocr_conf": 0.0, "match_method": "save_error"})

            # write JSON per-stem
            json_path = out_dir / f"{stem}_candidates.json"
            try:
                info_for_json = {
                    "stem": stem,
                    "chosen": {"angle": int(variant.get("angle", 0)), "flip": variant.get("flip"), "score": float(variant.get("score", 0.0)), "orig_file": str(sub.name)},
                    "ocr": variant.get("ocr", [])
                }
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(info_for_json, jf, indent=2)
            except Exception:
                pass

        else:
            # sub is a folder path in the original workflow
            info = pick_best_candidates_from_folder(sub, use_ocr, ocr_reader, force_all=force_all, min_score=min_score, allow_flips=allow_flips)
            stem = info["stem"]
            print(f"\n[STEM] {stem}  (match method: {info['reason']})")
            if not info["candidates"]:
                print("  No candidates found. Skipping.")
                rows.append({"stem": stem, "chosen": "", "score": 0.0, "ocr_text": "", "ocr_conf": 0.0, "match_method": "no_candidates"})
                continue

            for c in info["candidates"]:
                print(f"  candidate: {c['file']:<60} score={c['score']:.3f} ocr_count={c['ocr_count']} sample={c['ocr_sample']} variant={c.get('chosen_variant')}")

            chosen = info["chosen"]
            if chosen is None:
                print("  No best candidate selected. Skipping.")
                rows.append({"stem": stem, "chosen": "", "score": 0.0, "ocr_text": "", "ocr_conf": 0.0, "match_method": "no_choice"})
                continue

            chosen_file = sub / chosen["file"]
            try:
                chosen_extra = info.get("chosen_extra", None)
                # Prefer the transformed BGR if present; otherwise read chosen_file
                transformed = None
                angle = 0
                flipmode = None
                if chosen_extra and chosen_extra.get("img_bgr") is not None:
                    transformed = chosen_extra["img_bgr"]
                    angle = int(chosen_extra.get("angle", 0))
                    flipmode = chosen_extra.get("flip", None)

                if transformed is None:
                    transformed = read_img_cv(chosen_file)

                out_img = out_dir / f"upright_{stem}.jpg"
                out_enh = out_dir / f"upright_{stem}_enh.jpg"
                cv2.imwrite(str(out_img), transformed)
                cv2.imwrite(str(out_enh), enhance_readability(transformed))

                ocr_text, ocr_conf = ("", 0.0)
                if info.get("chosen_ocr"):
                    ocr_text, ocr_conf = extract_best_numeric_from_ocr(info["chosen_ocr"])

                print(f"  [SAVED] chosen {chosen['file']} angle={angle} flip={flipmode} score={chosen['score']:.3f} ocr_text='{ocr_text}'")
                rows.append({"stem": stem, "chosen": str(out_img.name), "score": float(chosen["score"]), "ocr_text": ocr_text, "ocr_conf": float(ocr_conf), "match_method": info["reason"]})
            except Exception as e:
                print("  [ERROR] saving chosen:", e)
                rows.append({"stem": stem, "chosen": "error", "score": float(chosen.get("score", 0.0)), "ocr_text": "", "ocr_conf": 0.0, "match_method": "save_error"})

            json_path = out_dir / f"{stem}_candidates.json"
            try:
                info_for_json = dict(info)
                if info_for_json.get("chosen_extra") and info_for_json["chosen_extra"].get("img_bgr") is not None:
                    ce = info_for_json["chosen_extra"].copy()
                    ce.pop("img_bgr", None)
                    info_for_json["chosen_extra"] = ce
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(info_for_json, jf, indent=2)
            except Exception:
                pass

    # CSV summary
    csv_path = csv_out if csv_out else (out_dir / "upright_final_log.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "chosen", "score", "ocr_text", "ocr_conf", "match_method"])
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in ["stem", "chosen", "score", "ocr_text", "ocr_conf", "match_method"]}
                w.writerow(row)
        print("\nWrote CSV:", csv_path)
    except Exception as e:
        print("Failed to write CSV:", e)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", required=True, help="root debug folder (subfolders or flat images)")
    p.add_argument("--out", default="upright_final", help="output folder")
    p.add_argument("--force-all", action="store_true", help="consider all images in subfolder (not strict stem-match)")
    p.add_argument("--no-ocr", action="store_true", help="disable OCR scoring (use orientation only)")
    p.add_argument("--no-flips", action="store_true", help="disable flipped variants")
    p.add_argument("--min-score", type=float, default=0.01)
    p.add_argument("--csv", default=None, help="optional path for CSV summary")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debug_root = Path(args.debug)
    if not debug_root.exists() or not debug_root.is_dir():
        print("Invalid debug root:", debug_root)
        sys.exit(1)

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    use_ocr = not args.no_ocr
    allow_flips = not args.no_flips
    if _HAS_EASYOCR:
        print("EasyOCR available.")
    else:
        print("EasyOCR not available.")
    if _HAS_PYTESS:
        print("pytesseract available.")
    else:
        print("pytesseract not available.")

    process_debug_root(debug_root, out_dir, force_all=args.force_all, use_ocr=use_ocr, allow_flips=allow_flips, min_score=args.min_score, csv_out=Path(args.csv) if args.csv else None)
