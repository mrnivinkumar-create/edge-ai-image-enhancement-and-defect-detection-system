import os
import random
import shutil
from pathlib import Path

# Change this path to where your NEU-CLS images are
SOURCE = Path(r"C:\Users\ELCOT\Downloads\NEU-CLS\NEU-CLS")

DEST = Path("dataset")

classes = {
    "Cr_": "crazing",
    "In_": "inclusion",
    "Pa_": "patches",
    "PS_": "pitted_surface",
    "RS_": "rolled_in_scale",
    "Sc_": "scratches"
}

# Create folders
for split in ["train", "validation", "test"]:
    for cls in classes.values():
        (DEST / split / cls).mkdir(parents=True, exist_ok=True)

for prefix, cls in classes.items():

    images = list(SOURCE.glob(f"{prefix}*.bmp"))

    random.shuffle(images)

    total = len(images)

    train = int(total * 0.70)
    val = int(total * 0.15)

    train_images = images[:train]
    val_images = images[train:train + val]
    test_images = images[train + val:]

    for img in train_images:
        shutil.copy(img, DEST / "train" / cls / img.name)

    for img in val_images:
        shutil.copy(img, DEST / "validation" / cls / img.name)

    for img in test_images:
        shutil.copy(img, DEST / "test" / cls / img.name)

print("Dataset prepared successfully!")