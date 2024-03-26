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
        
    #print('mean score =', mean_score)
            
    return img_fin, mean_score

def load_image_select_channel(tif_image_name): 
    try:
        tif_image_path = os.path.join("./data/test_satellite/",tif_image_name)
        
        tif_img = imread(tif_image_path)

        ####################################################################################################################################
        
        #gamma = 1.5  # Adjust the gamma value as needed
        tif_img_ch_1, mean_score_ch_1 = normalize_img(tif_img[:, :, 1],type='images')
        #tif_img_ch_1 = np.int32(cv2.pow(tif_img_ch_1 / 255.0, gamma) * 255.0)
        
        tif_img_ch_0, mean_score_ch_0 = normalize_img(tif_img[:, :, 0],type='images')
        #tif_img_ch_0 = np.int32(cv2.pow(tif_img_ch_0 / 255.0, gamma) * 255.0)
        
        channel_diff = np.abs(tif_img_ch_1 * (mean_score_ch_0/mean_score_ch_1) - tif_img_ch_0)
        
        
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
     
        ####################################################################################################################################
 
    except Exception as e:
        print("Error:", e)
        image_img = None
        
    return image_img


# Load the model and weights
model = tf.keras.models.load_model('CNN_model_v3.h5')
model.load_weights('CNN_model_weights_v3.h5')

# Assuming 'test_images' is a list of paths to test set images
metadata = pd.read_csv("./data/metadata.csv")
test_images = metadata[metadata['dataset'] == 'test_img']['filename'].values

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
    #predicted_image = predicted_image[0, :, :]  # Remove batch and channel dimensions
  

    # Add a 0-edge to make it (350, 350)
    predicted_image_with_edge = np.pad(predicted_image, ((1, 1), (1, 1)), mode='constant')
    print(np.sum(predicted_image_with_edge))


    # Save the resulting image to the results folder with a similar name to the input image
    
    # Extract the base name without the extension
    base_name = os.path.splitext(test_image_name)[0]
    # Replace the suffix
    kelp_filename = base_name.replace("_satellite", "_kelp") + ".tif"
    cv2.imwrite(os.path.join('results', kelp_filename), predicted_image_with_edge)
