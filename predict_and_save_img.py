import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tifffile import imread

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

def load_image_select_channel(test_image_name): 
    try:
        test_image_path = os.path.join("./train_val_test_data/test_images/",test_image_name)
        
        tif_img = imread(test_image_path)
        tif_img_slc = tif_img[:, :, 1:2]
        
        image_img = normalize_img(img=tif_img_slc, type="images")
        
    except Exception as e:
        print("Error:", e)
        
    return image_img


# Load the model and weights
model = tf.keras.models.load_model('CNN_model_v1.h5')
model.load_weights('CNN_model_weights_v1.h5')

# Assuming 'test_images' is a list of paths to test set images
#metadata = pd.read_csv("./data/metadata_new.csv")
#test_images = metadata[metadata['dataset'] == 'test_img']['filename'].values
test_images = sorted(os.listdir("./train_val_test_data/test_images/"))

# Create results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Predict on test set images
for test_image_name in test_images:
    # Load the test set image
    
    test_image = load_image_select_channel(test_image_name)
    #test_image = np.expand_dims(test_image, axis=-1)  # Add channel dimension

    # Predict
    predicted_image = model.predict(np.expand_dims(test_image, axis=0))

    # Post-processing
    predicted_image = (predicted_image > 0.5).astype(np.uint8)  # Binary thresholding
    predicted_image = predicted_image[0, :, :, 0]  # Remove batch and channel dimensions
    

    # Add a 0-edge to make it (350, 350)
    predicted_image_with_edge = np.pad(predicted_image, ((1, 1), (1, 1)), mode='constant')


    # Save the resulting image to the results folder with a similar name to the input image
    
    # Extract the base name without the extension
    base_name = os.path.splitext(test_image_name)[0]
    # Replace the suffix
    kelp_filename = base_name.replace("_satellite", "_kelp") + ".tif"
    cv2.imwrite(os.path.join('results', kelp_filename), predicted_image_with_edge)
