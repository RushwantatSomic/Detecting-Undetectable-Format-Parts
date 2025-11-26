import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(src, dst, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    classes = ["0", "90", "180", "270"]

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dst, split, cls), exist_ok=True)

    for cls in classes:
        class_path = os.path.join(src, cls)
        if not os.path.isdir(class_path):
            print(f"Warning: folder not found: {class_path}")
            continue

        images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not images:
            print(f"No images found in {class_path}")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1 - val_size_adjusted, random_state=42)

        for img in train_imgs:
            shutil.copy(img, os.path.join(dst, "train", cls))

        for img in val_imgs:
            shutil.copy(img, os.path.join(dst, "val", cls))

        for img in test_imgs:
            shutil.copy(img, os.path.join(dst, "test", cls))

        print(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder containing 0/90/180/270")
    parser.add_argument("--dst", required=True, help="Destination dataset folder")
    args = parser.parse_args()

    split_dataset(args.src, args.dst)
    print("Dataset created successfully!")
