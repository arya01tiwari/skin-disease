import os
import shutil
import random

# Path to the original dataset
original_dataset_dir = "Skin Cancer Dataset"
base_dir = "dataset"  # output folder

# Split ratio
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Create dataset/train, val, test dirs
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_dir, split)
    os.makedirs(split_path, exist_ok=True)

# For each class folder
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(train_split * len(images))
    val_end = int((train_split + val_split) * len(images))

    split_sets = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, split_images in split_sets.items():
        split_class_dir = os.path.join(base_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img_name in split_images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(split_class_dir, img_name)
            shutil.copyfile(src, dst)

print("✅ Dataset split into train/val/test!")
