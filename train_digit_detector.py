"""
YOLOv8 Digit Detector Training Script
=====================================
Dataset: Nov27FormatParts v2
Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Instructions:
1. Extract your Roboflow dataset zip file
2. Place this script in the SAME folder as the extracted dataset
   OR update the DATASET_PATH below
3. Run: python train_digit_detector.py
"""

from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================
# CONFIGURATION - Update these paths if needed
# ============================================================

# Path to your data.yaml file (relative or absolute)
# If script is in same folder as extracted dataset:
DATASET_YAML = "data.yaml"

# Or use absolute path:
# DATASET_YAML = r"C:\Users\YourName\Downloads\Nov27FormatParts-2\data.yaml"

# Model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
# Recommended: 'n' for real-time, 's' for better accuracy
MODEL_SIZE = "n"

# Training parameters
EPOCHS = 100          # Number of training epochs
BATCH_SIZE = 16       # Reduce to 8 if GPU memory error
IMAGE_SIZE = 640      # Match your dataset
PATIENCE = 20         # Early stopping patience

# Output
PROJECT_NAME = "runs/detect"
EXPERIMENT_NAME = "digit_detector_v1"

# ============================================================
# TRAINING SCRIPT - No need to modify below
# ============================================================

def train():
    print("="*60)
    print("YOLOv8 Digit Detector Training")
    print("="*60)
    
    # Verify dataset exists
    if not os.path.exists(DATASET_YAML):
        print(f"ERROR: Cannot find data.yaml at: {DATASET_YAML}")
        print("Please update DATASET_YAML path in this script.")
        return None
    
    print(f"Dataset: {DATASET_YAML}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print("="*60)
    
    # Load pretrained YOLOv8 model
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    
    # Train
    results = model.train(
        # Dataset
        data=DATASET_YAML,
        
        # Training params
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        
        # Optimizer
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Augmentation (minimal - already done in Roboflow)
        hsv_h=0.01,        # Slight hue variation
        hsv_s=0.3,         # Saturation variation
        hsv_v=0.3,         # Value variation
        degrees=3.0,       # Small rotation (Roboflow already did ±5)
        translate=0.1,     # Translation
        scale=0.2,         # Scale variation
        flipud=0.0,        # NO vertical flip (critical!)
        fliplr=0.0,        # NO horizontal flip (critical!)
        mosaic=0.5,        # Mosaic augmentation
        mixup=0.0,         # No mixup
        
        # Output
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        
        # Hardware
        device=0,          # GPU 0 (use 'cpu' if no GPU)
        workers=8,         # Data loading workers
        
        # Misc
        verbose=True,
        plots=True,
        save=True,
        save_period=10,    # Save checkpoint every 10 epochs
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    best_model_path = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
    last_model_path = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "last.pt"
    
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    print(f"\nTraining plots saved to: {Path(PROJECT_NAME) / EXPERIMENT_NAME}")
    
    return results


def validate():
    """Run validation on the trained model"""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    best_model_path = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"ERROR: Model not found at {best_model_path}")
        print("Please run training first.")
        return None
    
    model = YOLO(str(best_model_path))
    
    # Validate on test set
    metrics = model.val(
        data=DATASET_YAML,
        split="test",
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"mAP50:    {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
    print(f"mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
    
    print("\nPer-class AP50:")
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i, ap in enumerate(metrics.box.ap50):
        status = "✓" if ap > 0.90 else "⚠" if ap > 0.80 else "✗"
        print(f"  Digit {class_names[i]}: {ap:.4f} ({ap*100:.1f}%) {status}")
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    else:
        # Train the model
        results = train()
        
        if results:
            # Run validation after training
            print("\nRunning validation on test set...")
            validate()
