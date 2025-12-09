# orient_and_fix.py (robust, Windows-friendly)
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageOps
import tempfile, shutil, sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEG_MAP = {"0": 0, "90": 90, "180": 180, "270": 270}

def extract_top_and_conf(result):
    # Works across Ultralytics versions
    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs
        top_idx = int(probs.top1)
        # try data tensor first
        try:
            conf = float(probs.data[top_idx])
            return top_idx, conf
        except Exception:
            pass
        try:
            conf = float(probs.top1conf)
            return top_idx, conf
        except Exception:
            pass
        try:
            conf = float(probs[top_idx])
            return top_idx, conf
        except Exception:
            pass
    # fallback: if names exist and single-class
    try:
        if hasattr(result, "boxes") and len(result.boxes):
            conf = float(result.boxes.conf[0].item()) if hasattr(result.boxes, "conf") else 0.0
            cls = int(result.boxes.cls[0].item()) if hasattr(result.boxes, "cls") else 0
            return cls, conf
    except Exception:
        pass
    return 0, 0.0

def top_idx_to_degrees(result, top_idx):
    # Prefer reading class name if available
    try:
        if hasattr(result, "names") and result.names is not None:
            # names may be e.g. {0: "0", 1: "90", ...}
            name = result.names[top_idx]
            # if name is numeric string like "90"
            if isinstance(name, str) and name.strip().isdigit():
                return int(name.strip())
    except Exception:
        pass
    # fallback: map index 0->0,1->90,2->180,3->270
    return [0,90,180,270][int(top_idx) % 4]

def rotate_ccw_to_upright(img, deg_clockwise):
    # deg_clockwise = how many degrees the image is rotated clockwise now.
    # To undo we rotate counter-clockwise by same amount.
    deg_ccw = deg_clockwise % 360
    if deg_ccw == 0:
        return img
    return img.rotate(deg_ccw, expand=True)

def predict_and_fix(model, src_path, dst_path, device, accept_conf):
    # predict
    # ultralytics expects a path, we pass string
    reslist = model.predict(str(src_path), device=device, verbose=False)
    if not reslist:
        return False, "no_result"
    r = reslist[0]
    top_idx, conf = extract_top_and_conf(r)
    deg = top_idx_to_degrees(r, top_idx)
    # Accept or not
    if conf < accept_conf:
        # if accept_conf==0.0 we always accept
        return False, f"low_conf:{conf:.3f}"
    # load image with EXIF transpose
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img)
    fixed = rotate_ccw_to_upright(img, deg)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # preserve format/quality for common formats
    fmt = None
    if dst_path.suffix.lower() in (".jpg", ".jpeg"):
        fmt = "JPEG"
        fixed.save(dst_path, format=fmt, quality=95)
    else:
        fixed.save(dst_path)
    return True, f"fixed_by_pred_{deg}_conf{conf:.3f}"

def process_recursive(model, source, output, device="cpu", accept_conf=0.7):
    source = Path(source)
    output = Path(output)
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} images under {source}")
    total = 0
    fixed = 0
    skipped = 0
    for p in sorted(files):
        total += 1
        rel = p.relative_to(source)
        dst = output / rel
        try:
            ok, reason = predict_and_fix(model, p, dst, device, accept_conf)
            if ok:
                fixed += 1
                print(f"[OK] {p} -> {dst} ({reason})")
            else:
                # if accept_conf==0.0 we will never reach here because we accept all
                skipped += 1
                # still write original to output to ensure "ALL_UPRIGHT" contains a file
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
                print(f"[SKIP] {p} -> copied original ({reason})")
        except Exception as e:
            skipped += 1
            print(f"[ERR] {p} -> {e}")
    print(f"Done. total={total} fixed={fixed} skipped={skipped} out={output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--accept-conf", type=float, default=0.7, help="confidence threshold; 0.0 accepts all")
    args = parser.parse_args()

    model = YOLO(args.weights)
    process_recursive(model, args.source, args.output, device=args.device, accept_conf=args.accept_conf)

if __name__ == "__main__":
    main()
