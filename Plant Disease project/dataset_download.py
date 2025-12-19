import kagglehub
import os
import shutil
import random

# Download dataset
dataset_root = kagglehub.dataset_download("emmarex/plantdisease")
print("Dataset downloaded to:", dataset_root)

# Actual images are inside PlantVillage/
DATASET_DIR = os.path.join(dataset_root, "PlantVillage")

OUTPUT_DIR = "PlantVillage_Split"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create output directories
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Loop through disease folders
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print(" Dataset downloaded and split successfully.")
