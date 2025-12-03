from ultralytics import YOLO
import os
import cv2

# ============================================================
# CONFIGURATION FOR SIZE FILTERING (like your colleague's)
# ============================================================

# Confidence threshold for inference (not used in training itself)
CONFIDENCE_THRESHOLD = 0.50  # 50%

# Relative size filter: keep boxes whose height >= X% of max height
SIZE_RATIO_THRESHOLD = 0.50  # 50% of largest height

# Absolute minimum pixel height: remove tiny detections
MIN_PIXEL_HEIGHT = 15


# ============================================================
# SIZE FILTERING HELPER
# ============================================================

def filter_by_size_from_result(
    result,
    size_ratio_threshold: float = SIZE_RATIO_THRESHOLD,
    min_height: int = MIN_PIXEL_HEIGHT,
):
    """
    Filter detections in a single YOLO 'result' object based on bounding box height.
    This mirrors your colleague's logic but works directly on Ultralytics Results.

    Steps:
    1. Remove boxes whose height < min_height (absolute filter)
    2. Compute max height among remaining boxes
    3. Keep only boxes whose height >= size_ratio_threshold * max_height
    """
    boxes = result.boxes  # ultralytics.yolo.engine.results.Boxes

    # No detections
    if boxes is None or len(boxes) == 0:
        return result

    # xyxy shape: [N, 4]
    xyxy = boxes.xyxy  # (x1, y1, x2, y2)

    # Compute heights (as tensor)
    heights = xyxy[:, 3] - xyxy[:, 1]

    # 1) Absolute height filter
    keep = heights >= min_height

    # If everything is filtered out
    if keep.sum() == 0:
        result.boxes = boxes[keep]  # empty Boxes object
        return result

    # 2) Relative to max height among remaining
    max_h = heights[keep].max()
    min_acceptable = max_h * size_ratio_threshold

    keep = heights >= min_acceptable

    # Apply mask to boxes (Boxes supports indexing)
    result.boxes = boxes[keep]

    return result


# ============================================================
# OPTIONAL: INFERENCE WITH SIZE FILTER (for drawing boxes)
# ============================================================

def run_inference_with_size_filter(
    model,
    source,
    output_folder: str = "inference_with_size_filter",
    conf: float = CONFIDENCE_THRESHOLD,
):
    """
    Run inference on 'source' and save images where:
    - YOLO detections are filtered by confidence and size
    - Only filtered boxes are drawn on the saved images

    'source' can be:
        - path to image
        - path to folder
        - video, etc. (same as Ultralytics API)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Run YOLO inference
    results = model(source, conf=conf)

    # Handle single or multiple results
    if not isinstance(results, list):
        results = [results]

    for i, res in enumerate(results):
        # Apply size-based filtering to this result
        res = filter_by_size_from_result(res)

        # Draw only the filtered boxes
        annotated = res.plot()

        # Build output path
        img_name = f"result_{i}.jpg"
        out_path = os.path.join(output_folder, img_name)

        # Save annotated image
        cv2.imwrite(out_path, annotated)
        print(f"Saved filtered result to: {out_path}")


# ============================================================
# TRAINING (YOUR ORIGINAL PART, UNCHANGED)
# ============================================================

def main():
    # Load a pretrained YOLOv8 model (choose nano/small/medium/etc.)
    model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt, yolov8m.pt...

    # Train the model
    model.train(
        data="data.yaml",   # path to your data.yaml
        epochs=100,         # adjust as needed
        imgsz=640,          # training image size
        batch=16,           # adjust depending on your GPU
        workers=8,
        device="cuda"
    )

    # Optional: evaluate on test set (standard validation)
    model.val()

    # --------------------------------------------------------
    # Optional: run a quick inference with size filtering
    # (Uncomment and set your test image/folder if you want)
    # --------------------------------------------------------
    # test_source = "path/to/your/test/image_or_folder"
    # run_inference_with_size_filter(model, test_source)


if __name__ == "__main__":
    main()
