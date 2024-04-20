import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
import cv2
import numpy as np
import os


def normalize_img(img, type):
    try:
        img = np.array(img)
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        
        if min_val == max_val:
            img_fin = np.zeros_like(img)
        else:
            if type == "images":
                # Map to [0,1] for neural networks
                img_fin = (img - min_val) / (max_val - min_val)
            elif type == "labels":
                # Map to 0 and 1 for labels
                threshold = 0.5
                img_fin = (img > threshold).astype(np.uint8)
            
        
    except ValueError as ve:
        print("ValueError:", ve)
        img_fin = None
        
    except Exception as e:
        print("Error:", e)
        img_fin = None
            
    return img_fin


def load_image_select_channel(tif_image_path, kelp_image_path): 
    try:
        tif_img = imread(tif_image_path)
        # Select channel 1 (Near infrared cuz it shows the most kelp)
        tif_RGB_img = tif_img[:, :, 1:2]
        
        kelp_img = cv2.imread(kelp_image_path, cv2.IMREAD_GRAYSCALE)
        kelp_RGB_img = np.expand_dims(kelp_img, axis=-1)
        
        image_img = normalize_img(img=tif_RGB_img, type="images")
        labels_img = normalize_img(img=kelp_RGB_img, type="labels")
        
    except Exception as e:
        print("Error:", e)
        
    return image_img, labels_img


def make_full_tensor(images_folder, labels_folder):
    full_images = []
    full_labels = []

    image_files = sorted(os.listdir(images_folder))
    label_files = sorted(os.listdir(labels_folder))

    for image_name, label_name in zip(image_files, label_files):
        tif_image_path = os.path.join(images_folder, image_name)
        kelp_image_path = os.path.join(labels_folder, label_name)
        tif_img, label_img = load_image_select_channel(tif_image_path, kelp_image_path)
        full_images.append(tif_img)
        full_labels.append(label_img)

    full_images = np.array(full_images)
    full_labels = np.array(full_labels)
    
    return full_images, full_labels



# Define paths to train, validation, and test folders
train_images_folder = "./data/train_val_test_data/train_images/"
train_labels_folder = "./data/train_val_test_data/train_labels/"

val_images_folder = "./data/train_val_test_data/val_images/"
val_labels_folder = "./data/train_val_test_data/val_labels/"

test_images_folder = "./data/train_val_test_data/test_images/"
test_labels_folder = "./data/train_val_test_data/test_labels/"

# Create preprocessed data directory
preprocessed_data_dir = "./preprocessed_data/"
os.makedirs(preprocessed_data_dir, exist_ok=True)

# Make full tensors for train data
full_train_images, full_train_labels = make_full_tensor(train_images_folder, train_labels_folder)
print("Size check")
print("full train imgs and labels shape", full_train_images.shape, full_train_labels.shape)
np.save(os.path.join(preprocessed_data_dir, "full_train_images.npy"), full_train_images)
np.save(os.path.join(preprocessed_data_dir, "full_train_labels.npy"), full_train_labels)

# Make full tensors for validation data
full_val_images, full_val_labels = make_full_tensor(val_images_folder, val_labels_folder)
print("full val imgs and labels shape", full_val_images.shape, full_val_labels.shape)
np.save(os.path.join(preprocessed_data_dir, "full_val_images.npy"), full_val_images)
np.save(os.path.join(preprocessed_data_dir, "full_val_labels.npy"), full_val_labels)

# Make full tensors for test data
full_test_images, full_test_labels = make_full_tensor(test_images_folder, test_labels_folder)
print("full test imgs and labels shape", full_test_images.shape, full_test_labels.shape)
np.save(os.path.join(preprocessed_data_dir, "full_test_images.npy"), full_test_images)
np.save(os.path.join(preprocessed_data_dir, "full_test_labels.npy"), full_test_labels)

print("Data preprocessing completed and saved as numpy arrays.")
