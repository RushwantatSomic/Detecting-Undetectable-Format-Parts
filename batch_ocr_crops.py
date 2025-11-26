# batch_ocr_crops_ensemble_fixed.py
"""
Batch OCR with multi-variant preprocessing, ensemble voting, and conservative geometry fixes.
Saves debug images in <CROP_FOLDER>/ocr_debug_images and ROI fixes in <CROP_FOLDER>/ocr_debug_images/fix_rois.
"""

import os
import sys
import csv
import traceback
from pathlib import Path
from PIL import Image
import cv2
import pytesseract
import shutil
import numpy as np

# ML helpers
import torch
import torchvision.transforms as T

# Optional dependencies
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')  # CPU by default
except Exception:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
except Exception:
    EASYOCR_AVAILABLE = False

# Prefer skimage thinning if available (fast)
USE_SKIMAGE_THIN = False
try:
    from skimage.morphology import skeletonize
    USE_SKIMAGE_THIN = True
except Exception:
    USE_SKIMAGE_THIN = False

# path to your trained digit classifier (TorchScript recommended)
DIGIT_CLS_PATH = "digit_cnn_best.pt"

# try load if exists
digit_model = None
if os.path.exists(DIGIT_CLS_PATH):
    try:
        digit_model = torch.jit.load(DIGIT_CLS_PATH, map_location="cpu")
        digit_model.eval()
        print("Digit classifier loaded:", DIGIT_CLS_PATH)
    except Exception as e:
        print("Failed to load digit classifier:", e)
        digit_model = None

# simple transform: resize to 28x28 and normalise
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def classify_digit_roi(roi_np):
    """
    roi_np: numpy image (grayscale or color). Returns (digit_str, confidence_percent).
    """
    if digit_model is None:
        return None, 0.0
    try:
        # ensure HWC uint8
        if isinstance(roi_np, Image.Image):
            roi_np = np.array(roi_np)
        if roi_np.ndim == 2:
            inp = roi_np  # grayscale
        else:
            inp = cv2.cvtColor(roi_np, cv2.COLOR_BGR2GRAY)
        inp_t = transform(inp).unsqueeze(0)  # shape [1,1,28,28]
        with torch.no_grad():
            out = digit_model(inp_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            cls = int(probs.argmax())
            conf = float(probs.max()) * 100.0
            return str(cls), conf
    except Exception as e:
        return None, 0.0

# --------- Config (edit as needed) ----------
CROP_FOLDER = r"C:\Users\Gnanasekar\Downloads\MODELS\Digit_Upright\ALL_UPRIGHT"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # or None
# --------------------------------------------

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OUT_CSV = os.path.join(CROP_FOLDER, "ocr_debug.csv")
OUT_DEBUG_DIR = os.path.join(CROP_FOLDER, "ocr_debug_images")
OUT_FIX_ROIS = os.path.join(OUT_DEBUG_DIR, "fix_rois")

# recreate debug folders
if os.path.exists(OUT_DEBUG_DIR):
    shutil.rmtree(OUT_DEBUG_DIR)
os.makedirs(OUT_DEBUG_DIR, exist_ok=True)
os.makedirs(OUT_FIX_ROIS, exist_ok=True)

def list_files(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    return files

def simple_read_test(path):
    img = cv2.imread(path)
    if img is None:
        return False, "cv2.imread returned None"
    try:
        im = Image.open(path)
        im.verify()
    except Exception as e:
        return False, f"PIL verify failed: {e}"
    return True, "ok"

# ---------------- preprocessing ----------------

def deskew_image(gray):
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess(img_bgr, target_char_px=40):
    """
    Return list of preprocessing variants: list of (variant_name, image_np).
    Each image_np is a single-channel (grayscale) uint8 image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # deskew
    gray_ds = deskew_image(gray_clahe)

    # scale small images up
    scale = 2 if max(h, w) < 200 else 1
    gray_ds = cv2.resize(gray_ds, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray_ds, (3,3), 0)

    # threshold variants
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    th_adapt_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
    th_adapt_gauss = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    th_adapt_mean_inv = cv2.bitwise_not(th_adapt_mean)
    th_adapt_gauss_inv = cv2.bitwise_not(th_adapt_gauss)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    def clean(im):
        return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=1)

    candidates = [
        ("gray", gray_ds),
        ("otsu", clean(th_otsu)),
        ("inv_otsu", clean(th_inv_otsu)),
        ("adapt_mean", clean(th_adapt_mean)),
        ("adapt_gauss", clean(th_adapt_gauss)),
        ("adapt_mean_inv", clean(th_adapt_mean_inv)),
        ("adapt_gauss_inv", clean(th_adapt_gauss_inv)),
    ]
    return candidates

# ---------------- OCR helpers ----------------

def try_tesseract(img_pil, psm=7, whitelist="0123456789"):
    try:
        cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(img_pil, config=cfg)
        text = ''.join(ch for ch in text if ch.isdigit())
        return text, 0.0
    except Exception:
        return "", 0.0

def parse_conf_list(conf_list):
    out = []
    for c in conf_list:
        try:
            cf = float(c)
            if cf >= 0:
                out.append(cf)
        except Exception:
            continue
    return out

def try_tesseract_data(img_pil, psm=7, oem=3, whitelist="0123456789"):
    """
    Returns digits, avg_conf (0..100), full_text, raw_data_dict
    """
    try:
        cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
        data = pytesseract.image_to_data(img_pil, config=cfg, output_type=pytesseract.Output.DICT)
        texts = data.get('text', [])
        confs = data.get('conf', [])
        txt = " ".join(t for t in texts if t.strip() != "")
        digits = ''.join(ch for ch in txt if ch.isdigit())
        confs_num = parse_conf_list(confs)
        avg_conf = float(sum(confs_num) / len(confs_num)) if confs_num else 0.0
        return digits, avg_conf, txt, data
    except Exception:
        return "", 0.0, "", {}

def try_easyocr(img_np):
    if not EASYOCR_AVAILABLE:
        return "", 0.0
    try:
        res = easyocr_reader.readtext(img_np, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        best = max(res, key=lambda x: x[2])
        txt = ''.join(ch for ch in best[1] if ch.isdigit())
        conf = best[2] * 100.0
        return txt, conf
    except Exception:
        return "", 0.0

def ocr_candidates_from_image(img_bgr):
    """
    Try PaddleOCR first (per-line + per-char confidences), then Tesseract/EasyOCR ensemble across variants.
    Returns a dict with winner and all_results list of tuples:
    (key, digits, avg_conf_0_100, full_text, raw_data, variant_name, variant_image)
    """
    results = []

    # PaddleOCR (if available)
    if PADDLE_AVAILABLE:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            paddle_res = paddle_reader.ocr(img_rgb, cls=True)
            for i, line in enumerate(paddle_res):
                box, (text, conf) = line
                digits = ''.join(ch for ch in text if ch.isdigit())
                avg_conf = float(conf) * 100.0 if conf is not None else 0.0
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x0, x1 = max(0, min(xs)), min(img_bgr.shape[1], max(xs))
                y0, y1 = max(0, min(ys)), min(img_bgr.shape[0], max(ys))
                try:
                    variant_img = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                except Exception:
                    variant_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                results.append((f"paddle_line_{i}", digits, avg_conf, text, {}, "paddle", variant_img))
        except Exception:
            pass

    # Generate preprocessing variants and run Tesseract/EasyOCR on each
    variants = preprocess(img_bgr)
    for vname, vimg in variants:
        # vimg is grayscale np.uint8
        # Tesseract OCR basic
        try:
            pil = Image.fromarray(vimg)
            digits1 = try_tesseract(pil, psm=7)[0]
            if digits1:
                results.append((f"tess_{vname}", digits1, 0.0, digits1, {}, f"tess_{vname}", vimg))
            # Tesseract data for confidences
            digits2, avg_conf2, fulltxt2, raw2 = try_tesseract_data(pil, psm=7)
            if digits2:
                results.append((f"tess_data_{vname}", digits2, float(avg_conf2), fulltxt2, raw2, f"tess_data_{vname}", vimg))
        except Exception:
            pass
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                txt, conf = try_easyocr(vimg)
                if txt:
                    results.append((f"easy_{vname}", txt, float(conf), txt, {}, f"easy_{vname}", vimg))
            except Exception:
                pass

    # Voting: collect by digit string
    vote = {}
    for key, digits, avg_conf, full_txt, raw_data, variant_name, variant_img in results:
        if not digits:
            continue
        # normalize conf: ensure numeric 0..100
        if avg_conf is None:
            avg_conf = 0.0
        avg_conf = float(avg_conf)
        rec = vote.get(digits)
        # boost Paddle results if present
        score = avg_conf
        if key.startswith("paddle"):
            score *= 1.05
        if rec is None:
            vote[digits] = [1, score, key, variant_img, avg_conf]
        else:
            rec[0] += 1
            if score > rec[1]:
                rec[1] = score
                rec[2] = key
                rec[3] = variant_img
                rec[4] = avg_conf

    if vote:
        winner_text, meta = max(vote.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        count, best_score, best_key, best_variant_img, best_conf = meta
        return {
            "key": best_key,
            "digits": winner_text,
            "avg_conf": float(best_conf),
            "full_text": winner_text,
            "raw_data": {},
            "variant_name": "ensemble",
            "transform_name": "",
            "variant_image": best_variant_img,
            "all_results": results
        }

    # fallback empty
    return {"key":"none","digits":"","avg_conf":0.0,"full_text":"","raw_data":{}, "variant_name":"", "transform_name":"", "variant_image":None, "all_results":[]}

# --------------- geometry-based helpers ---------------

def zhang_suen_thinning(img):
    # Prefer skimage skeletonize if available
    if USE_SKIMAGE_THIN:
        bw = (img > 0).astype(np.uint8)
        sk = skeletonize(bw).astype(np.uint8) * 255
        return sk
    # fallback: simple morphological thinning-like filter (fast-ish)
    # Using OpenCV ximgproc.thinning if available
    try:
        th = cv2.ximgproc.thinning(img)
        return th
    except Exception:
        # last fallback: binary erosion until no change (not true thinning but ok as fallback)
        bw = (img > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        prev = None
        while True:
            er = cv2.erode(bw, kernel, iterations=1)
            sub = cv2.subtract(bw, er)
            bw = er.copy()
            if prev is not None and np.array_equal(bw, prev):
                break
            prev = bw.copy()
        return bw

def count_skeleton_endpoints(skel):
    if skel is None or skel.size == 0:
        return 0
    skel_bin = (skel > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.int32)
    conv = cv2.filter2D(skel_bin, -1, kernel)
    pts = np.where(skel_bin == 1)
    count = 0
    for y,x in zip(pts[0], pts[1]):
        neigh = conv[y,x] - 10
        if neigh == 1:
            count += 1
    return count

def compute_features_for_roi(roi):
    f = {}
    img = roi.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(bw == 255) < np.sum(bw == 0):
        bw = cv2.bitwise_not(bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        hflat = hierarchy[0]
        holes = sum(1 for h in hflat if int(h[3]) >= 0)
    f['holes'] = holes

    ys, xs = np.where(bw == 255)
    if len(xs) == 0:
        f['aspect_ratio'] = 0.0
        f['solidity'] = 0.0
        f['centroid_y_rel'] = 0.5
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        f['aspect_ratio'] = float(w) / float(h + 1e-9)
        M = cv2.moments(bw)
        cy = (M['m01'] / (M['m00'] + 1e-9)) if M['m00'] != 0 else (y0 + h/2)
        f['centroid_y_rel'] = (cy - y0) / (h + 1e-9)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            f['solidity'] = area / (hull_area + 1e-9)
        else:
            f['solidity'] = 0.0

    pad = 2
    pad_img = cv2.copyMakeBorder(bw, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    skel = zhang_suen_thinning(pad_img)
    endpoints = count_skeleton_endpoints(skel)
    f['endpoints'] = endpoints

    hroi, wroi = bw.shape
    left = bw[:, :wroi//2].astype(np.int32)
    right = bw[:, (wroi - wroi//2):].astype(np.int32)
    if left.shape[1] != right.shape[1]:
        minw = min(left.shape[1], right.shape[1])
        left = left[:, :minw]; right = right[:, :minw]
    rev_right = np.fliplr(right)
    v_diff = np.mean(np.abs(left - rev_right)) / 255.0
    f['vertical_sym'] = 1.0 - v_diff

    top = bw[:hroi//2, :].astype(np.int32)
    bottom = bw[(hroi - hroi//2):, :].astype(np.int32)
    if top.shape[0] != bottom.shape[0]:
        minh = min(top.shape[0], bottom.shape[0])
        top = top[:minh, :]; bottom = bottom[-minh:, :]
    rev_bottom = np.flipud(bottom)
    h_diff = np.mean(np.abs(top - rev_bottom)) / 255.0
    f['horizontal_sym'] = 1.0 - h_diff

    top_band = bw[:max(1, hroi//4), :]; mid_band = bw[hroi//3:2*hroi//3, :]; bot_band = bw[-max(1, hroi//4):, :]
    f['mid_density'] = float(np.mean(mid_band == 255))
    f['top_density'] = float(np.mean(top_band == 255))
    f['bot_density'] = float(np.mean(bot_band == 255))

    return f

def classify_digit_from_features(f, ocr_hint=None):
    """
    Safer geometry-based digit guesser.
    Returns (digit_str, confidence_float_in_0_1) or (None, 0.0) when unsure.
    Conservative: avoids defaulting to '8' unless very strong evidence.
    """
    holes = int(f.get('holes', 0))
    ar = float(f.get('aspect_ratio', 0.5))
    solidity = float(f.get('solidity', 0.5))
    cy = float(f.get('centroid_y_rel', 0.5))
    endpoints = int(f.get('endpoints', 0))
    v_sym = float(f.get('vertical_sym', 0.5))
    h_sym = float(f.get('horizontal_sym', 0.5))
    mid_density = float(f.get('mid_density', 0.0))
    top_density = float(f.get('top_density', 0.0))
    bot_density = float(f.get('bot_density', 0.0))

    # Strong hole evidence -> 8 (very confident)
    if holes >= 2 and solidity > 0.5 and v_sym > 0.5:
        return '8', 0.98

    # Single hole: prefer 0/6/9 with conservative thresholds
    if holes == 1:
        if v_sym > 0.75 and 0.6 <= ar <= 1.4 and solidity > 0.45:
            return '0', 0.92
        if cy > 0.57 and solidity > 0.4:
            return '6', 0.86
        if cy < 0.43 and solidity > 0.4:
            return '9', 0.86
        return None, 0.0

    # No holes: use endpoints & density heuristics conservatively
    if endpoints == 0:
        # Only accept '8' if many signals agree
        if mid_density > 0.30 and v_sym > 0.65 and solidity > 0.6:
            return '8', 0.88
        return None, 0.0

    if ar < 0.30 and endpoints <= 2:
        return '1', 0.94

    if top_density > 0.28 and v_sym < 0.5 and endpoints >= 2 and ar > 0.45:
        return '7', 0.88

    if mid_density > 0.22 and endpoints == 2 and v_sym < 0.6:
        return '3', 0.88

    if mid_density < 0.22 and top_density > bot_density and endpoints >= 2:
        return '5', 0.82

    if endpoints >= 2 and cy < 0.5 and mid_density < 0.4:
        return '2', 0.78

    # fallback: only use OCR hint (low confidence) if present
    if ocr_hint and ocr_hint.isdigit():
        return ocr_hint, 0.45

    return None, 0.0

def geometry_fix_all_digits(variant_img, ocr_text, fname_for_debug=None):
    """
    Segment variant image into digit ROIs and apply geometry classifier.
    Returns (fixed_text, fixes) where fixes is list of (index, original, pred, pred_conf).
    """
    img = variant_img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(bw == 255) < np.sum(bw == 0):
        bw = cv2.bitwise_not(bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bw_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 30:
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return ocr_text, []

    boxes = sorted(boxes, key=lambda b: b[0])
    digits = list(ocr_text)
    if len(boxes) != len(digits):
        proj = np.sum(bw_clean == 255, axis=0)
        splits = []
        threshold = np.max(proj) * 0.05
        inside = False
        start = 0
        for i, v in enumerate(proj):
            if v > threshold and not inside:
                inside = True
                start = i
            if v <= threshold and inside:
                inside = False
                splits.append((start, i))
        boxes = []
        for s,e in splits:
            if e - s < 3:
                continue
            boxes.append((s, 0, e - s, img.shape[0]))
        if len(boxes) != len(digits):
            boxes = []
            total_w = img.shape[1]
            n = max(1, len(digits))
            step = total_w / n
            for i in range(n):
                x = int(i * step)
                w = int(step)
                boxes.append((x, 0, w, img.shape[0]))

    fixed = list(digits)
    fixes = []
    for i, box in enumerate(boxes[:len(digits)]):
        x,y,w,h = box
        pad = max(1, int(0.05 * max(w,h)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
        roi = img[y0:y1, x0:x1]
        feats = compute_features_for_roi(roi)
        original = digits[i] if i < len(digits) else None
        pred, conf = classify_digit_from_features(feats, ocr_hint=original)
        apply_change = False

        if pred is None:
            apply_change = False
        else:
            if original is None:
                if conf >= 0.92:
                    apply_change = True
            else:
                if pred == original:
                    apply_change = False
                else:
                    risky_pairs = {('3','8'), ('8','3'), ('3','0'), ('0','3'), ('6','8'), ('9','8')}
                    if (original, pred) in risky_pairs:
                        if conf >= 0.96:
                            apply_change = True
                    else:
                        if conf >= 0.92:
                            apply_change = True

        if apply_change and pred is not None:
            fixes.append((i, original, pred, conf))
            fixed[i] = pred
            try:
                if fname_for_debug:
                    roi_name = os.path.join(OUT_FIX_ROIS, f"{fname_for_debug}_idx{i}_{original}_to_{pred}.png")
                else:
                    roi_name = os.path.join(OUT_FIX_ROIS, f"fix_idx{i}_{original}_to_{pred}.png")
                cv2.imwrite(roi_name, roi)
            except Exception:
                pass

    return "".join(fixed), fixes

def generate_transform_variants(img):
    """
    Given a single-channel (grayscale) image numpy array, return a list of
    (transform_name, transformed_img) including rotations and flips.
    """
    variants = []
    if len(img.shape) == 3:
        base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        base = img.copy()
    base = base.astype(np.uint8)

    variants.append(("R0", base))
    variants.append(("R90", cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)))
    variants.append(("R180", cv2.rotate(base, cv2.ROTATE_180)))
    variants.append(("R270", cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    variants.append(("FH", cv2.flip(base, 1)))
    variants.append(("FV", cv2.flip(base, 0)))
    variants.append(("FH_R90", cv2.rotate(cv2.flip(base, 1), cv2.ROTATE_90_CLOCKWISE)))

    seen = set()
    unique = []
    for name, v in variants:
        key = (v.shape, v.tobytes()[:64])
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, v))
    return unique

def color_mask_and_crop(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = []
    lower1 = np.array([0,60,40]); upper1 = np.array([10,255,255])
    lower2 = np.array([170,60,40]); upper2 = np.array([180,255,255])
    masks.append(cv2.inRange(hsv, lower1, upper1))
    masks.append(cv2.inRange(hsv, lower2, upper2))
    lowerb = np.array([95,60,40]); upperb = np.array([135,255,255])
    masks.append(cv2.inRange(hsv, lowerb, upperb))

    best = None
    best_area = 0
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if area > best_area and area > 50:
                best_area = area
                best = (x,y,w,h)
    if best is not None:
        x,y,w,h = best
        pad = 4
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(img_bgr.shape[1], x+w+pad); y1 = min(img_bgr.shape[0], y+h+pad)
        return img_bgr[y0:y1, x0:x1]
    return img_bgr

# --------------- main loop ---------------

def main():
    files = list_files(CROP_FOLDER)
    if not files:
        print("No image files found in:", CROP_FOLDER)
        sys.exit(1)

    rows = []
    print("Tesseract cmd:", getattr(pytesseract.pytesseract, 'tesseract_cmd', 'not set'))
    print("EasyOCR available:", EASYOCR_AVAILABLE)
    print("PaddleOCR available:", PADDLE_AVAILABLE)
    print("Files found:", len(files))

    for fname in files:
        full = os.path.join(CROP_FOLDER, fname)
        ok, msg = simple_read_test(full)
        debug_img_paths = []
        if not ok:
            rows.append([fname, "", 0.0, "error", msg])
            print(fname, "READ ERROR:", msg)
            continue
        try:
            img = cv2.imread(full)
            base = os.path.splitext(fname)[0]

            # save original preview
            try:
                orig_p = os.path.join(OUT_DEBUG_DIR, base + "_orig.png")
                cv2.imwrite(orig_p, img)
                debug_img_paths.append(orig_p)
            except Exception:
                pass

            # save preprocessing variants (small set)
            variants = preprocess(img)
            for vname, vimg in variants:
                try:
                    p = os.path.join(OUT_DEBUG_DIR, f"{base}_{vname}.png")
                    cv2.imwrite(p, vimg)
                    debug_img_paths.append(p)
                except Exception:
                    pass

            # OCR & ensemble
            ocr_res = ocr_candidates_from_image(img)
            all_results = ocr_res.get("all_results", [])

            # Voting logic already performed inside ocr_candidates_from_image; use result
            chosen_text = ocr_res.get("digits", "")
            chosen_conf = float(ocr_res.get("avg_conf", 0.0))
            chosen_key = ocr_res.get("key", "none")
            chosen_variant_img = ocr_res.get("variant_image", None)

            if not chosen_text and all_results:
                best_single = max(all_results, key=lambda r: (len(r[1]), r[2]))
                chosen_text = best_single[1]
                chosen_conf = best_single[2]
                chosen_key = best_single[0]
                chosen_variant_img = best_single[6]

            if chosen_variant_img is None:
                chosen_variant_img = variants[0][1] if variants else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Conservative geometry fixes:
            final_text = chosen_text
            final_conf = chosen_conf
            if any(ch.isdigit() for ch in chosen_text):
                fixed_text, geometry_fixes = geometry_fix_all_digits(chosen_variant_img, chosen_text, fname_for_debug=base)
                if geometry_fixes:
                    max_geom_conf = max(f[3] for f in geometry_fixes)
                    apply = False
                    if max_geom_conf >= 0.88:
                        apply = True
                    elif chosen_conf < 25.0 and max_geom_conf >= 0.75:
                        apply = True
                    if apply and fixed_text != chosen_text:
                        print(f"Applying geometry fixes for {fname}: {geometry_fixes} ({chosen_text} -> {fixed_text})")
                        final_text = fixed_text
                        final_conf = max(chosen_conf, max_geom_conf * 100.0)
                    else:
                        final_text = chosen_text
                        final_conf = chosen_conf
                else:
                    final_text = chosen_text
                    final_conf = chosen_conf
            else:
                final_text = chosen_text
                final_conf = chosen_conf

            # annotate and save
            ann = img.copy()
            label = f"{final_text} ({int(final_conf)}) [{chosen_key}]"
            cv2.putText(ann, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            ann_p = os.path.join(OUT_DEBUG_DIR, base + "_ann.png")
            cv2.imwrite(ann_p, ann)
            debug_img_paths.append(ann_p)

            rows.append([fname, final_text, final_conf, chosen_key, ";".join(debug_img_paths)])
            print(fname, "->", final_text, final_conf, chosen_key)

        except Exception as e:
            rows.append([fname, "", 0.0, "exception", str(e)])
            print("Exception for", fname)
            traceback.print_exc()

    # save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","detected_digits","confidence","engine","debug_images"])
        w.writerows(rows)

    print("Done. Debug CSV:", OUT_CSV)
    print("Debug images saved in:", OUT_DEBUG_DIR)
    print("ROI fixes saved in:", OUT_FIX_ROIS)

if __name__ == "__main__":
    main()
