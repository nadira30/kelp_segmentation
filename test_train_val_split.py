import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import os

# Load metadata CSV file
metadata = pd.read_csv("./data/metadata_new.csv")

# Filter training images and labels
train_images = metadata[metadata['dataset'] == 'train_img']['filename'].values
train_labels = metadata[metadata['dataset'] == 'label_img']['filename'].values

# Split training set into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1765, random_state=42) # 15 / (70 + 15)

# Ensure the lengths match
assert len(train_images) == len(train_labels)
assert len(val_images) == len(val_labels)
assert len(test_images) == len(test_labels)

# Create directories for storing original data
original_data_dir = "./train_val_test_data/"
os.makedirs(original_data_dir, exist_ok=True)

train_images_dir = os.path.join(original_data_dir, "train_images")
train_labels_dir = os.path.join(original_data_dir, "train_labels")
val_images_dir = os.path.join(original_data_dir, "val_images")
val_labels_dir = os.path.join(original_data_dir, "val_labels")
test_images_dir = os.path.join(original_data_dir, "test_images")
test_labels_dir = os.path.join(original_data_dir, "test_labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Copy original data files to corresponding directories
for image_name, label_name in zip(train_images, train_labels):
    shutil.copy(os.path.join("./data/train_satellite/", image_name), train_images_dir)
    shutil.copy(os.path.join("./data/train_kelp/", label_name), train_labels_dir)

for image_name, label_name in zip(val_images, val_labels):
    shutil.copy(os.path.join("./data/train_satellite/", image_name), val_images_dir)
    shutil.copy(os.path.join("./data/train_kelp/", label_name), val_labels_dir)

for image_name, label_name in zip(test_images, test_labels):
    shutil.copy(os.path.join("./data/train_satellite/", image_name), test_images_dir)
    shutil.copy(os.path.join("./data/train_kelp/", label_name), test_labels_dir)
