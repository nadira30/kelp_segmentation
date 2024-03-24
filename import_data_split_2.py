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
        img = img.astype(np.float64)
        min_val = np.min(img)
        max_val = np.max(img)
        
        if min_val == max_val:
            img_fin = np.zeros_like(img)
            mean_score = 0.01
        else:
            if type == "images":
                # Map to [0,1] for neural networks
                img_fin = (img - min_val) / (max_val - min_val)
                mean_score = np.sum(img_fin)/(350*350)
            elif type == "labels":
                # Map to 0 and 1 for labels
                threshold = 0.5
                img_fin = (img > threshold).astype(np.uint8)
                mean_score = np.sum(img_fin)/(350*350)
                
            
    except ValueError as ve:
        print("ValueError:", ve)
        img_fin = None
        
    except Exception as e:
        print("Error:", e)
        img_fin = None
            
    return img_fin, mean_score

def load_image_select_channel(tif_image_name,kelp_image_name): 
    try:
        tif_image_path = os.path.join("./data/train_satellite/",tif_image_name)
        kelp_image_path = os.path.join("./data/train_kelp/",kelp_image_name)
        
        tif_img = imread(tif_image_path)
        #tif_RGB_img = tif_img[:, :, 1:2]
        
        ####################################################################################################################################
        #------------experimental--------------------------
        #tif_RGB_img = tif_img[:, :, 1]-tif_img[:, :, 0]
        # Normalize the image array
        
        gamma = 1.5  # Adjust the gamma value as needed
        tif_img_ch_1, mean_score_ch_1 = normalize_img(tif_img[:, :, 1],type='images')
        #tif_img_ch_1 = np.int32(cv2.pow(tif_img_ch_1 / 255.0, gamma) * 255.0)
        
        tif_img_ch_0, mean_score_ch_0 = normalize_img(tif_img[:, :, 0],type='images')
        #tif_img_ch_0 = np.int32(cv2.pow(tif_img_ch_0 / 255.0, gamma) * 255.0)
        
        channel_diff = np.abs(tif_img_ch_1 * (mean_score_ch_0/mean_score_ch_1) - tif_img_ch_0)
        channel_diff = channel_diff
        
        #print('=======channel diff1===============')
        #print(channel_diff)
        #print('=======channel diff2===============')
        channel_diff,_ = normalize_img(channel_diff,type='images')
        #print(channel_diff)

        # Set a threshold for the absolute difference
        threshold = 20.0/255.0 # Adjust this threshold as needed
        
        # Create a mask where the absolute difference is below the threshold
        image_img = np.where(channel_diff < threshold, 0, tif_img_ch_1)
        #tif_RGB_img = tif_RGB_img.astype('int32')
        
        
        #------------experimental--------------------------
        kelp_img = cv2.imread(kelp_image_path, cv2.IMREAD_GRAYSCALE)        
        labels_img,_ = normalize_img(img=kelp_img, type="labels")
     
        ####################################################################################################################################
        
        #kelp_img = cv2.imread(kelp_image_path, cv2.IMREAD_GRAYSCALE        
        #image_img = normalize_img(img=tif_RGB_img, type="images")
        #labels_img = normalize_img(img=kelp_RGB_img, type="labels")
        
        
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
metadata = pd.read_csv("./data/metadata.csv")

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


