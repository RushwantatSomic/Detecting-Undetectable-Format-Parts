#!/usr/bin/env python3
"""
Debug runner for ocr_v4 pipeline.
- prints detailed progress
- logs exceptions to debug_log.txt
- creates output dirs and writes CSV even if empty
- writes a small "run_status.txt" with summary
"""

import os, sys, traceback, csv
import cv2
import pytesseract
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
CROP_FOLDER = r"C:\Users\Gnanasekar\Downloads\number detection"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OUT_ROOT = os.path.join(CROP_FOLDER, "ocr_v4_results")
ROI_FOLDER = os.path.join(OUT_ROOT, "rois")
CSV_PATH = os.path.join(OUT_ROOT, "ocr_v4_results.csv")
DIAG_CSV = os.path.join(OUT_ROOT, "ocr_v4_diagnostics.csv")
LOG_PATH = os.path.join(OUT_ROOT, "debug_log.txt")
RUN_STATUS = os.path.join(OUT_ROOT, "run_status.txt")
# ----------------------------

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(ROI_FOLDER, exist_ok=True)

# set tesseract path if available
if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    print("WARNING: Tesseract binary not found at configured path:", TESSERACT_CMD)
    print("If Tesseract is installed elsewhere, update TESSERACT_CMD in the script.")

# simple helpers (copied minimal versions from v4)
def list_files(folder):
    exts = ('.png','.jpg','.jpeg','.PNG','.JPG','.JPEG')
    return sorted([f for f in os.listdir(folder) if f.endswith(exts)])

def crop_color_tag(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = []
    masks.append(cv2.inRange(hsv, np.array([0,60,50]), np.array([12,255,255])))
    masks.append(cv2.inRange(hsv, np.array([170,60,50]), np.array([180,255,255])))
    masks.append(cv2.inRange(hsv, np.array([90,60,50]), np.array([135,255,255])))
    full = masks[0]
    for m in masks[1:]:
        full = cv2.bitwise_or(full, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, False
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = 4
    x0 = max(0, x-pad); y0 = max(0, y-pad); x1 = min(img.shape[1], x+w+pad); y1 = min(img.shape[0], y+h+pad)
    return img[y0:y1, x0:x1], True

def ensure_fg_white(th):
    if np.sum(th==255) < np.sum(th==0):
        return cv2.bitwise_not(th)
    return th

def variants_from_gray(gray):
    variants = {}
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['otsu'] = ensure_fg_white(otsu)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        _, cotsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants['clahe_otsu'] = ensure_fg_white(cotsu)
    except Exception:
        pass
    try:
        adp = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,10)
        variants['adaptive_mean'] = ensure_fg_white(adp)
    except Exception:
        pass
    return variants

def tesseract_conf_for_image(img, psm=7):
    cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    confs=[]
    for txt, conf in zip(data.get('text', []), data.get('conf', [])):
        if txt is None or str(txt).strip()=="":
            continue
        try:
            cf = float(conf)
        except Exception:
            continue
        if any(ch.isdigit() for ch in str(txt)):
            confs.append(cf)
    if confs:
        return confs, sum(confs)/len(confs)
    return [], 0.0

# Main debug run
files = list_files(CROP_FOLDER)
with open(LOG_PATH, "w", encoding="utf-8") as logf:
    try:
        logf.write("Debug run start\n")
        logf.write("CROP_FOLDER: {}\n".format(CROP_FOLDER))
        logf.write("Found {} files\n".format(len(files)))
        print("Found", len(files), "image files in", CROP_FOLDER)
        if len(files) == 0:
            print("No images found. Check file extensions and path.")
            logf.write("No input images - exiting.\n")
        rows=[]
        diag=[]
        for i, fname in enumerate(files):
            try:
                print(f"({i+1}/{len(files)}) Processing:", fname)
                logf.write(f"Processing: {fname}\n")
                full = os.path.join(CROP_FOLDER, fname)
                img = cv2.imread(full)
                if img is None:
                    print("  -> cv2.imread returned None (file may be corrupted).")
                    logf.write("  -> READ FAIL\n")
                    rows.append([fname, "READ_FAIL", "", "", "0.000%"])
                    continue
                crop, tag_found = crop_color_tag(img)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                variants = variants_from_gray(gray)
                # quick diagnostic of variants
                vn = list(variants.keys())
                logf.write("  variants: " + ",".join(vn) + "\n")
                # try whole-image best conf as simple check
                best_text = ""
                best_avg = 0.0
                best_variant = None
                for vname, bin_img in variants.items():
                    confs, avg = tesseract_conf_for_image(bin_img, psm=7)
                    txt = ''.join(ch for ch in pytesseract.image_to_string(bin_img, config="--psm 7 -c tessedit_char_whitelist=0123456789") if ch.isdigit())
                    logf.write(f"    variant {vname}: text='{txt}' avg_conf={avg:.3f} conf_count={len(confs)}\n")
                    if (len(txt) > 0 and avg >= best_avg) or (len(txt) > len(best_text)):
                        best_text = txt
                        best_avg = avg
                        best_variant = vname
                print(f"  best_variant={best_variant} text='{best_text}' avg={best_avg:.3f}")
                # save crop and a binary preview from best variant (if exists)
                if best_variant:
                    preview = variants[best_variant]
                    cv2.imwrite(os.path.join(OUT_ROOT, Path(fname).stem + "_preview.png"), preview)
                cv2.imwrite(os.path.join(OUT_ROOT, Path(fname).stem + "_crop.png"), crop)
                rows.append([fname, best_text, best_variant or "", f"{best_avg:.3f}%"])
            except Exception as e:
                tb = traceback.format_exc()
                print("  ERROR processing", fname)
                print(tb)
                logf.write("ERROR:\n" + tb + "\n")
                rows.append([fname, "ERROR", "", "0.000%"])
        # write CSVs even if empty
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["filename","detected","variant","avg_conf"])
            w.writerows(rows)
        with open(DIAG_CSV, "w", newline="", encoding="utf-8") as fdiag:
            w = csv.writer(fdiag)
            w.writerow(["filename","detected","variant","avg_conf"])
            w.writerows(diag)
        logf.write("Run finished OK\n")
        print("Run finished. Wrote CSV to:", CSV_PATH)
        with open(RUN_STATUS, "w", encoding="utf-8") as s:
            s.write("OK\n")
            s.write("files_processed: {}\n".format(len(files)))
            s.write("csv: {}\n".format(CSV_PATH))
    except Exception as e:
        tb = traceback.format_exc()
        print("Fatal error during run:")
        print(tb)
        logf.write("FATAL:\n" + tb + "\n")
        with open(RUN_STATUS, "w", encoding="utf-8") as s:
            s.write("ERROR\n")
            s.write(tb)
