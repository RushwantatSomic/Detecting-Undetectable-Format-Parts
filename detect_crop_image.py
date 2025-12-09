import cv2
from pathlib import Path
from ultralytics import YOLO


def crop_from_image(model_path, image_path, output_folder, conf=0.2):
    # Load model
    print(f"Loading detector: {model_path}")
    model = YOLO(model_path)

    # Prepare paths
    image_path = Path(image_path)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print("ERROR: Cannot read image:", image_path)
        return

    # Run YOLO prediction
    print("Running detection...")
    results = model.predict(str(image_path), conf=conf, verbose=False)
    result = results[0]

    if not len(result.boxes):
        print("No detections found.")
        return

    # Crop each bbox
    print(f"Found {len(result.boxes)} detections. Saving crops...")
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_path = out_dir / f"crop_{i+1:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        print("Saved:", crop_path)

    print("Done!")


if __name__ == "__main__":
    MODEL = r"C:\Users\Gnanasekar\Downloads\processtest1\models\detect\best.pt"
    IMAGE = r"C:\Users\Gnanasekar\Downloads\processtest1\inputs\images\20250814_112126185_iOS_jpg.rf.2b3ab7cb3c6e10a170efc889bd7f76ad.jpg"
    OUTPUT = r"C:\Users\Gnanasekar\Downloads\processtest1\outputs\crops"

    crop_from_image(MODEL, IMAGE, OUTPUT, conf=0.1)
