#!/usr/bin/env python3
"""
crop_tags_with_detector.py

Use a trained detector (YOLOv8 via 'ultralytics' or a generic PyTorch .pt) to find format-part tags
in each input image, crop them and save into a flat output folder, plus CSV.

Usage examples:
    # YOLOv8 usage (recommended)
    python crop_tags_with_detector.py --input INPUT_DIR --output OUT_DIR --yolo-model path/to/best.pt --class-name formatpart

    # Generic PyTorch detection model (must output boxes, scores, labels)
    python crop_tags_with_detector.py --input INPUT_DIR --output OUT_DIR --torch-model path/to/model.pt --class-index 0
"""

import os
import cv2
import csv
import argparse
from pathlib import Path
import numpy as np

# ---------- Configurable defaults ----------
DEFAULT_CONF_THRESH = 0.35   # detector score threshold
FALLBACK_TO_HSV = True       # if detector finds nothing, optionally use HSV mask
# -------------------------------------------

# Try to import ultralytics (YOLOv8). If not present, the script will still work with PyTorch mode.
_HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

# PyTorch imports for generic model loading
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# ---------- Helpers ----------
def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_image(p):
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return im

def color_mask_red_blue(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,50,50]), np.array([10,255,255]))
    m2 = cv2.inRange(hsv, np.array([170,50,50]), np.array([180,255,255]))
    m3 = cv2.inRange(hsv, np.array([85,40,40]), np.array([145,255,255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.bitwise_or(mask, m3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 3)
    return mask

def boxes_from_mask(mask, min_area=500):
    # return axis-aligned boxes from contours on mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        out.append((x,y,x+w,y+h, float(area)))
    return out

# ---------- Detector wrapper classes ----------
class YOLODetector:
    def __init__(self, model_path, class_name=None, conf_thresh=DEFAULT_CONF_THRESH):
        if not _HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics package not found. Install with `pip install ultralytics` to use YOLO mode.")
        self.model = YOLO(model_path)
        # build mapping: if model has class names, we'll use them.
        self.class_name = class_name
        self.conf_thresh = conf_thresh
        # try to get names from model
        try:
            nm = getattr(self.model.model, 'names', None)
            if nm:
                # nm is dict index->name
                self.names = nm
            else:
                self.names = {}
        except Exception:
            self.names = {}

    def predict(self, img_bgr):
        """
        Run model and return list of detections as (x1,y1,x2,y2,score,class_id,class_name)
        """
        # ultralytics accepts OpenCV BGR directly
        preds = self.model.predict(source=img_bgr, imgsz=1280, conf=self.conf_thresh, verbose=False)
        # preds is a list per image; we passed a single image
        out = []
        try:
            boxes = preds[0].boxes  # ultralytics Boxes object
            for b in boxes:
                conf = float(b.conf.cpu().numpy()) if hasattr(b, 'conf') else float(b.conf)
                cls = int(b.cls.cpu().numpy()) if hasattr(b, 'cls') else int(b.cls)
                xyxy = b.xyxy.cpu().numpy().astype(int).tolist() if hasattr(b, 'xyxy') else list(map(int, b.xyxy[0].tolist()))
                x1,y1,x2,y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                name = self.names.get(cls, str(cls))
                out.append((x1,y1,x2,y2,conf,cls,name))
        except Exception:
            # older ultralytics variations: parse preds[0].boxes.xyxy etc.
            try:
                det = preds[0].boxes.xyxy.numpy()   # Nx4
                confs = preds[0].boxes.conf.numpy()
                clsids = preds[0].boxes.cls.numpy().astype(int)
                for (x1,y1,x2,y2), conf, cls in zip(det, confs, clsids):
                    name = self.names.get(int(cls), str(int(cls)))
                    if conf >= self.conf_thresh:
                        out.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls), name))
            except Exception:
                pass
        # optionally filter by requested class_name
        if self.class_name is not None and len(out) > 0:
            filtered = [d for d in out if d[6] == self.class_name or d[5] == self.class_name or str(d[5]) == str(self.class_name)]
            if filtered:
                out = filtered
        return out

class TorchDetector:
    def __init__(self, model_path, conf_thresh=DEFAULT_CONF_THRESH, device='cpu'):
        if not _HAS_TORCH:
            raise RuntimeError("torch not found. Install PyTorch to use generic torch model.")
        # tries to load generic torch model - you may need to adapt this loader for state_dict models
        self.device = torch.device(device)
        try:
            # try torch.jit or full model
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.is_jit = True
        except Exception:
            try:
                # load state_dict into a user model is not implemented here (you must provide a scriptable model)
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                self.is_jit = False
            except Exception as e:
                raise RuntimeError(f"Failed to load torch model: {e}")
        self.conf_thresh = conf_thresh

    def predict(self, img_bgr):
        """
        Expect the model to return a dict/list with keys/tensors 'boxes','scores','labels'
        boxes: Nx4 in xyxy
        scores: N
        labels: N (ints)
        Returns list similar to YOLODetector
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # convert to tensor normalized [0,1], shape (1,C,H,W)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)
        with torch.no_grad():
            out = self.model(img_t)
        # 'out' parsing must be adapted to your model; attempt common formats
        detections = []
        try:
            # if model returns list[dict] like torchvision models
            if isinstance(out, (list, tuple)):
                o = out[0]
            else:
                o = out
            boxes = o.get('boxes', None) or getattr(o, 'boxes', None)
            scores = o.get('scores', None) or getattr(o, 'scores', None)
            labels = o.get('labels', None) or getattr(o, 'labels', None)
            if boxes is None or scores is None:
                # try to interpret out as tensor Nx6 (x1,y1,x2,y2,score,class)
                if isinstance(o, torch.Tensor) and o.ndim == 2 and o.shape[1] >= 6:
                    arr = o.cpu().numpy()
                    for row in arr:
                        x1,y1,x2,y2,score,cls = row[:6]
                        if score >= self.conf_thresh:
                            detections.append((int(x1),int(y1),int(x2),int(y2), float(score), int(cls), str(int(cls))))
                    return detections
                return []
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy() if labels is not None else [0]*len(scores)
            for (x1,y1,x2,y2), s, lbl in zip(boxes, scores, labels):
                if float(s) >= self.conf_thresh:
                    detections.append((int(x1),int(y1),int(x2),int(y2), float(s), int(lbl), str(int(lbl))))
        except Exception:
            # unknown format
            return []
        return detections

# ---------- Main pipeline ----------
def crop_with_detector(input_dir, output_dir, detector=None, use_detector=True, class_filter=None, conf_thresh=DEFAULT_CONF_THRESH, fallback_hsv=FALLBACK_TO_HSV, min_area_hsv=600):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mkdirp(output_dir)
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')])
    rows = []
    global_idx = 0

    for p in images:
        img = read_image(p)
        detections = []
        if use_detector and detector is not None:
            try:
                detections = detector.predict(img)
            except Exception as e:
                print(f"[WARN] Detector failed on {p.name}: {e}")
                detections = []

        # optionally filter by class_filter if provided (string or int)
        if class_filter is not None and detections:
            filtered = []
            for d in detections:
                # d = (x1,y1,x2,y2,conf,cls,name)
                # Accept if name matches or cls matches numeric index
                if str(d[6]) == str(class_filter) or str(d[5]) == str(class_filter) or (isinstance(class_filter, str) and class_filter.lower() in str(d[6]).lower()):
                    filtered.append(d)
            if filtered:
                detections = filtered

        # If detector produced no detections and fallback requested -> use mask
        if (not detections) and fallback_hsv:
            mask = color_mask_red_blue(img)
            boxes = boxes_from_mask(mask, min_area=min_area_hsv)
            # convert to detection format (x1,y1,x2,y2,score,cls,name)
            detections = []
            for (x1,y1,x2,y2,area) in boxes:
                detections.append((x1,y1,x2,y2, 0.5, -1, "hsv"))

        # crop each detection
        for det in detections:
            global_idx += 1
            x1,y1,x2,y2,score,cls,name = det
            # clip coords
            h,w = img.shape[:2]
            x1c = max(0, min(w-1, int(x1)))
            y1c = max(0, min(h-1, int(y1)))
            x2c = max(0, min(w-1, int(x2)))
            y2c = max(0, min(h-1, int(y2)))
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = img[y1c:y2c, x1c:x2c].copy()
            raw_name = f"{p.stem}_g{global_idx:06d}_raw.png"
            out_raw = output_dir / raw_name
            cv2.imwrite(str(out_raw), crop)
            # try deskew with rotated rect around bbox points for slightly rotated boxes
            # compute minAreaRect from bbox contour
            rect_box = np.array([[x1c,y1c],[x2c,y1c],[x2c,y2c],[x1c,y2c]], dtype=np.int32)
            r_crop, rect_info, box_pts = deskew_crop_from_bbox(img, rect_box)
            if r_crop is None:
                deskew_name = f"{p.stem}_g{global_idx:06d}_deskew_failed.png"
                out_deskew = output_dir / deskew_name
                cv2.imwrite(str(out_deskew), crop)
            else:
                deskew_name = f"{p.stem}_g{global_idx:06d}_deskew.png"
                out_deskew = output_dir / deskew_name
                cv2.imwrite(str(out_deskew), r_crop)

            rows.append({
                "source": p.name,
                "global_index": global_idx,
                "raw_crop": raw_name,
                "deskew_crop": deskew_name,
                "x1": int(x1c),
                "y1": int(y1c),
                "x2": int(x2c),
                "y2": int(y2c),
                "score": float(score),
                "class": str(name),
            })

    # write CSV
    csv_path = output_dir / "detector_crops_log.csv"
    if rows:
        keys = list(rows[0].keys())
    else:
        keys = ["source","global_index","raw_crop","deskew_crop","x1","y1","x2","y2","score","class"]
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Saved", len(rows), "crops to", output_dir)
    print("CSV:", csv_path)

# ---------- utility: deskew from bbox (axis aligned box) ----------
def deskew_crop_from_bbox(img, box_pts, pad=8):
    """
    Given 4 points (x,y) in order, compute minAreaRect and deskew similar to previous script.
    Returns cropped deskew image or None.
    """
    cnt = box_pts.reshape((-1,1,2)).astype(np.int32)
    rect = cv2.minAreaRect(cnt)
    (cx,cy),(w,h),angle = rect
    if w <= 0 or h <= 0:
        return None, rect, None
    if w < h:
        w,h = h,w
        angle += 90.0
    box = cv2.boxPoints(((cx,cy),(w,h),angle))
    box_int = box.astype(np.int32)
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
    H,W = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (W,H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    box_rot = cv2.transform(np.array([box]), M)[0]
    xs = box_rot[:,0]; ys = box_rot[:,1]
    x0 = int(max(0, np.floor(xs.min()) - pad)); y0 = int(max(0, np.floor(ys.min()) - pad))
    x1 = int(min(W, np.ceil(xs.max()) + pad)); y1 = int(min(H, np.ceil(ys.max()) + pad))
    if x1 <= x0 or y1 <= y0:
        return None, rect, box_int
    crop = rotated[y0:y1, x0:x1].copy()
    return crop, rect, box_int

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Folder with main images")
    p.add_argument("--output", "-o", required=True, help="Folder to save crops")
    p.add_argument("--yolo-model", default=None, help="Path to YOLOv8 model (.pt) (ultralytics).")
    p.add_argument("--torch-model", default=None, help="Path to torch model (.pt) (generic).")
    p.add_argument("--class-name", default=None, help="Class name in YOLO model to pick (optional).")
    p.add_argument("--class-index", type=int, default=None, help="Class index to pick (generic torch mode).")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESH, help="Detector confidence threshold")
    p.add_argument("--fallback-hsv", action="store_true", default=False, help="Fallback to HSV mask if detector finds no boxes")
    p.add_argument("--min-area-hsv", type=int, default=600, help="Min area for HSV fallback")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mkdirp(args.output)

    detector = None
    use_detector = True
    if args.yolo_model:
        if not _HAS_ULTRALYTICS:
            raise SystemExit("ultralytics not installed. pip install ultralytics")
        print("Loading YOLO model:", args.yolo_model)
        detector = YOLODetector(args.yolo_model, class_name=args.class_name, conf_thresh=args.conf)
    elif args.torch_model:
        if not _HAS_TORCH:
            raise SystemExit("PyTorch not installed. pip install torch")
        print("Loading Torch model:", args.torch_model)
        detector = TorchDetector(args.torch_model, conf_thresh=args.conf)
    else:
        print("No detector model provided. Using HSV fallback only.")
        use_detector = False

    crop_with_detector(args.input, args.output, detector=detector, use_detector=use_detector, class_filter=(args.class_name if args.class_name else args.class_index), conf_thresh=args.conf, fallback_hsv=args.fallback_hsv, min_area_hsv=args.min_area_hsv)
