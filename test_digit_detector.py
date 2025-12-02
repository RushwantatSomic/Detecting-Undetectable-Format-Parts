"""
Test Inference Script for Digit Detector
=========================================
Use this script to test your trained model on sample images.

Usage:
    python test_digit_detector.py path/to/image.jpg
    python test_digit_detector.py path/to/folder/
"""

from ultralytics import YOLO
import cv2
import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Path to your trained model
MODEL_PATH = "runs/detect/digit_detector_v1/weights/best.pt"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Output folder for annotated images
OUTPUT_FOLDER = "test_results"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def filter_by_size(detections, size_ratio_threshold=0.5):
    """
    Filter out small digits (like serial numbers) based on bounding box height.
    Keeps only digits that are at least 50% of the maximum detected height.
    """
    if not detections:
        return []
    
    # Find maximum height
    max_height = max(d['height'] for d in detections)
    min_acceptable = max_height * size_ratio_threshold
    
    # Filter
    filtered = [d for d in detections if d['height'] >= min_acceptable]
    
    return filtered


def assemble_number(detections, tag_color=None):
    """
    Assemble detected digits into a number string.
    Sorts by x-coordinate (left to right).
    For blue tags, inserts hyphen after 2nd digit.
    """
    if not detections:
        return "", []
    
    # Sort by x-coordinate
    sorted_dets = sorted(detections, key=lambda d: d['x_center'])
    
    # Extract digit values
    digits = [str(d['digit']) for d in sorted_dets]
    
    # Assemble based on tag color
    if tag_color == "blue" and len(digits) == 4:
        # Format: XX-XX
        number = f"{digits[0]}{digits[1]}-{digits[2]}{digits[3]}"
    else:
        # Just concatenate
        number = "".join(digits)
    
    return number, sorted_dets


def process_image(model, image_path):
    """
    Process a single image and return detection results.
    """
    # Run inference
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            detections.append({
                'digit': class_id,
                'confidence': confidence,
                'x_center': (x1 + x2) / 2,
                'y_center': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1,
                'bbox': [x1, y1, x2, y2]
            })
    
    return detections, results[0] if results else None


def print_results(image_path, detections, number, filtered_count):
    """Print formatted results"""
    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    if not detections:
        print("No digits detected!")
        return
    
    print(f"Raw detections: {len(detections) + filtered_count}")
    print(f"After size filter: {len(detections)}")
    print(f"\nDetected digits (left to right):")
    
    for d in sorted(detections, key=lambda x: x['x_center']):
        print(f"  [{d['digit']}] conf: {d['confidence']:.2f}, "
              f"pos: ({d['x_center']:.0f}, {d['y_center']:.0f}), "
              f"size: {d['width']:.0f}x{d['height']:.0f}")
    
    print(f"\n>>> Assembled Number: {number}")


def save_annotated_image(result, image_path, output_folder):
    """Save image with detection boxes drawn"""
    os.makedirs(output_folder, exist_ok=True)
    
    annotated = result.plot()
    output_path = os.path.join(output_folder, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, annotated)
    
    return output_path


# ============================================================
# MAIN
# ============================================================

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python test_digit_detector.py <image_path_or_folder>")
        print("Example: python test_digit_detector.py test_image.jpg")
        print("Example: python test_digit_detector.py ./test_images/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first or update MODEL_PATH.")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!\n")
    
    # Get list of images to process
    if os.path.isdir(input_path):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = [
            os.path.join(input_path, f) 
            for f in os.listdir(input_path) 
            if f.lower().endswith(extensions)
        ]
        print(f"Found {len(image_paths)} images in folder")
    else:
        image_paths = [input_path]
    
    # Process each image
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"WARNING: File not found: {img_path}")
            continue
        
        # Detect digits
        detections, result = process_image(model, img_path)
        
        # Filter small digits
        original_count = len(detections)
        detections = filter_by_size(detections)
        filtered_count = original_count - len(detections)
        
        # Assemble number
        number, sorted_dets = assemble_number(detections)
        
        # Print results
        print_results(img_path, sorted_dets, number, filtered_count)
        
        # Save annotated image
        if result:
            output_path = save_annotated_image(result, img_path, OUTPUT_FOLDER)
            print(f"Annotated image saved: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"Processing complete! Results saved to: {OUTPUT_FOLDER}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
