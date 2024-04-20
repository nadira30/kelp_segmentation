import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
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

def load_image_select_channel(tif_image_name,kelp_image_name): 
    try:
        tif_image_path = os.path.join("./data/train_satellite/",tif_image_name)
        kelp_image_path = os.path.join("./data/train_kelp/",kelp_image_name)
        
        tif_img = imread(tif_image_path)
        # Select channel 1 (Near infrared cuz it shows the most kelp)
        tif_RGB_img = tif_img[:, :, 1:2]
        
        kelp_img = cv2.imread(kelp_image_path, cv2.IMREAD_GRAYSCALE)
        #kelp_RGB_img = cv2.cvtColor(kelp_img, cv2.COLOR_GRAY2RGB)
        kelp_RGB_img = np.expand_dims(kelp_img, axis=-1)
        
        image_img = normalize_img(img=tif_RGB_img, type="images")
        labels_img = normalize_img(img=kelp_RGB_img, type="labels")
        
        #print('tif img shape', tif_img.shape)
        #print('tif rgb img shape', tif_RGB_img.shape)
        #print('kelp img shape', kelp_img.shape) 
        #print('kelp rgb img shape', kelp_RGB_img.shape)
        
    except Exception as e:
        print("Error:", e)
        
    return image_img, labels_img

def make_full_tensor(images, labels):
    full_images = []
    full_labels = []

    for image_name, label_name in zip(images, labels):
        tif_img, label_img = load_image_select_channel(image_name, label_name)
        full_images.append(tif_img)
        full_labels.append(label_img)

    full_images = np.array(full_images)
    full_labels = np.array(full_labels)
    
    return full_images, full_labels

# Load metadata CSV file
metadata = pd.read_csv("/Users/nadira/gatech/Sp24/CV/kelp_segmentation/data/metadata_new.csv")

# Filter training images and labelst
train_images = metadata[metadata['dataset'] == 'train_img']['filename'].values
train_labels = metadata[metadata['dataset'] == 'label_img']['filename'].values
#test_images = metadata[metadata['dataset'] == 'test_img']['filename'].values

# Split training set into training and validation
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Ensure the lengths match
assert len(train_images) == len(train_labels)
assert len(val_images) == len(val_labels)

# Print the number of samples in each set
print(f"Number of training samples: {len(train_images)}")
print(f"Number of validation samples: {len(val_images)}")
#print(f"Number of test samples: {len(test_images)}")

full_train_images, full_train_labels =  make_full_tensor(train_images, train_labels)
full_val_images, full_val_labels =  make_full_tensor(val_images, val_labels)



print("Size check")
print("full train imgs and lebels shape", full_train_images.shape, full_train_labels.shape)
print("full val imgs and lebels shape", full_val_images.shape, full_val_labels.shape)


preprocessed_data_dir = "./preprocessed_data/"
os.makedirs(preprocessed_data_dir, exist_ok=True)

np.save(os.path.join(preprocessed_data_dir, "full_train_images.npy"), full_train_images)
np.save(os.path.join(preprocessed_data_dir, "full_train_labels.npy"), full_train_labels)

np.save(os.path.join(preprocessed_data_dir, "full_val_images.npy"), full_val_images)
np.save(os.path.join(preprocessed_data_dir, "full_val_labels.npy"), full_val_labels)


