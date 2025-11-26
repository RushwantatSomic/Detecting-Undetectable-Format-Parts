import os
import cv2
import pytesseract
import numpy as np
import csv
from PIL import Image

# ---------------- CONFIG ----------------
CROP_FOLDER = r"C:\Users\Gnanasekar\Downloads\MODELS\Digit_Upright\ALL_UPRIGHT"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OUT_FOLDER = os.path.join(CROP_FOLDER, "ocr_results_with_conf_3dp")
os.makedirs(OUT_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(OUT_FOLDER, "ocr_with_confidences_3dp.csv")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# -----------------------------------------

def crop_color_tag(img):
    """Extract red/blue tag from image automatically."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ranges = [
        # red lower
        (np.array([0,60,50]), np.array([10,255,255])),
        # red upper
        (np.array([170,60,50]), np.array([180,255,255])),
        # blue
        (np.array([90,60,50]), np.array([135,255,255]))
    ]

    full_mask = None
    for lo, hi in ranges:
        mask = cv2.inRange(hsv, lo, hi)
        full_mask = mask if full_mask is None else (full_mask | mask)

    if full_mask is None:
        return img

    # clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # find largest contour
    cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img

    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]


def preprocess(img):
    """Prepare for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # strong threshold
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ensure digits are dark on light background for Tesseract:
    if np.sum(th==0) > np.sum(th==255):
        th = cv2.bitwise_not(th)

    return th


def ocr_image_with_confidence(img):
    """
    Run OCR (Tesseract) and return:
      - digits: concatenated digits found (string)
      - confs: list of confidences for each digit-token (percent, floats)
      - raw_words: list of text tokens (useful for debugging)
    """
    # Use psm 7 (single line) and whitelist digits.
    config = r"--psm 7 -c tessedit_char_whitelist=0123456789"

    # plain OCR (string)
    text = pytesseract.image_to_string(img, config=config)
    digits = ''.join(ch for ch in text if ch.isdigit())

    # detailed output with confidences
    confs = []
    raw_words = []
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        n = len(data.get('text', []))
        for i in range(n):
            word = data['text'][i]
            conf = data['conf'][i]
            # Tesseract may return '-1' or non-numeric; filter those out.
            try:
                conff = float(conf)
            except Exception:
                continue
            # we only care about tokens that contain digits
            if word is None:
                continue
            word_digits = ''.join(ch for ch in str(word) if ch.isdigit())
            if word_digits:
                confs.append(conff)   # already 0..100 scale
                raw_words.append(word)
    except Exception:
        confs = []

    return digits, confs, raw_words


def main():
    files = [f for f in os.listdir(CROP_FOLDER) if f.lower().endswith((".jpg",".png",".jpeg"))]
    print("Files:", len(files))

    # prepare CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "detected_digits", "token_texts", "token_confidences", "avg_confidence"])

        for fname in files:
            full = os.path.join(CROP_FOLDER, fname)
            img = cv2.imread(full)
            if img is None:
                print(fname, "-> FAILED TO READ")
                writer.writerow([fname, "", "", "", "0.000%"])
                continue

            # 1. auto crop tag
            crop = crop_color_tag(img)

            # 2. preprocess
            prep = preprocess(crop)

            # 3. OCR + confidences
            digits, confs, raw_words = ocr_image_with_confidence(prep)

            # format confidences for printing (3 decimal places)
            if confs:
                confs_str = [f"{c:.3f}%" for c in confs]
                avg_conf = sum(confs) / len(confs)
                avg_conf_str = f"{avg_conf:.3f}%"
            else:
                confs_str = ["0.000%"]
                avg_conf = 0.0
                avg_conf_str = "0.000%"

            # save debug images (crop + preprocessed)
            base_name = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(OUT_FOLDER, f"{base_name}_crop.png"), crop)
            cv2.imwrite(os.path.join(OUT_FOLDER, f"{base_name}_prep.png"), prep)

            # print the line (digits + confidences + avg) similar to your example
            print(f"{fname} -> '{digits}'  confidences: {confs_str}  avg: {avg_conf_str}")

            # save csv row: token texts joined by ';', confidences joined by ';'
            writer.writerow([fname, digits, ";".join(raw_words), ";".join(confs_str) if confs else "", avg_conf_str])

    print("\nDone. CSV saved to:", CSV_PATH)


if __name__ == "__main__":
    main()
