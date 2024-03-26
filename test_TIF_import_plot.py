import matplotlib.pyplot as plt
from tifffile import imread
import cv2
import numpy as np


def normalize_img(img, type):
    try:
        img = np.array(img)
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        
        if min_val == max_val:
            img_fin = np.zeros_like(img)
            mean_score = 0.01
        else:
            if type == "images":
                # Map to [0,1] for neural networks
                img_fin = (img - min_val) / (max_val - min_val) * 255
                mean_score = np.sum(img_fin)/(350*350*255)
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

def show_7_ch_tif_image(image_path): # For 7 channel TIF images
    channel_type_list = [
        "Shortwave Infrared",
        "Near-Infrared",
        "Red",
        "Green",
        "Blue",
        "Cloud Mask",
        "Digital Elev Map\n(Land Mask)"
    ]
    
    try:
        # Open the TIF image
        img = imread(image_path)
                # Check the shape
        print('TIF sat img shape', img.shape)  # It should print (316, 316, 7)
        
        # Display each channel separately
        num_channels = img.shape[2]
        fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * 5, 5))
        for i in range(num_channels):
            img_to_plot,_ = normalize_img(img[:,:,i], type='images')
            axs[i].imshow(img_to_plot, cmap='gray')  # Assuming grayscale for each channel
            axs[i].axis('off')
            axs[i].set_title(f'Channel {i}:\n{channel_type_list[i]}')
        
        plt.show()
 
    except Exception as e:
        print("Error:", e)
        
def show_tif_image(image_path): # For 7 channel TIF images
    try:
        # Open the TIF image
        img = imread(image_path)
                # Check the shape
        print('TIF sat img shape', img.shape)  # It should print (316, 316, 7)
        
        rgb_img = np.array(img)
        rgb_img = img[:, :, 2:5]
        
        # Normalize the image array
        rgb_img_plot,_ = normalize_img(rgb_img, type='images')

        plt.figure()
        # Display the RGB image
        plt.imshow(rgb_img_plot)
        plt.axis('off')
        plt.show()
 
    except Exception as e:
        print("Error:", e)
        
def show_kelp_image(image_path): # For kelp images
    try:
        # Open the TIF image
        img = imread(image_path)
                # Check the shape
        print('kelp img shape', img.shape) 
        rgb_img = np.array(img)
        print('max = ',np.max(rgb_img))

        # Normalize the image array
        rgb_img_plot,_ = normalize_img(rgb_img, type='labels')
        
        plt.figure()
        plt.imshow(rgb_img_plot)
        plt.axis('off')
        plt.show()
 
    except Exception as e:
        print("Error:", e)
        
def show_both_overlay(tif_image_path,kelp_image_path): # For kelp images
    try:
        # Open the kelp image
        tif_img = imread(tif_image_path)
        tif_RGB_img = tif_img[:, :, 1]
        
        """
        #------------experimental--------------------------
        #tif_RGB_img = tif_img[:, :, 1]-tif_img[:, :, 0]
        # Normalize the image array
        
        gamma = 1.5  # Adjust the gamma value as needed
        tif_img_ch_1, mean_score_ch_1 = normalize_img(tif_img[:, :, 1], type='images')
        #tif_img_ch_1 = np.int32(cv2.pow(tif_img_ch_1 / 255.0, gamma) * 255.0)
        
        tif_img_ch_0, mean_score_ch_0 = normalize_img(tif_img[:, :, 0], type='images')
        #tif_img_ch_0 = np.int32(cv2.pow(tif_img_ch_0 / 255.0, gamma) * 255.0)
        
        channel_diff = np.abs(tif_img_ch_1 * (mean_score_ch_0/mean_score_ch_1) - tif_img_ch_0)
        channel_diff = channel_diff
        
        print('=======channel diff1===============')
        print(channel_diff)
        print('=======channel diff2===============')
        channel_diff,_ = normalize_img(channel_diff, type='images')
        channel_diff = channel_diff.astype(np.int32)
        print(channel_diff)

        # Set a threshold for the absolute difference
        threshold = 20.0 # Adjust this threshold as needed
        
        # Create a mask where the absolute difference is below the threshold
        tif_RGB_img = np.where(channel_diff < threshold, 0, tif_img_ch_1)
        #tif_RGB_img = tif_RGB_img.astype('int32')
        
        
        #------------experimental--------------------------
        """
        
        kelp_img = cv2.imread(kelp_image_path, cv2.IMREAD_GRAYSCALE)
        #kelp_RGB_img = cv2.cvtColor(kelp_img, cv2.COLOR_GRAY2RGB)
        #kelp_RGB_img = np.expand_dims(kelp_img, axis=-1)
        kelp_RGB_img,_ = normalize_img(kelp_img, type='labels')
        kelp_RGB_img = kelp_RGB_img.astype(np.int32)
        
        print('tif img shape', tif_img.shape, tif_img.dtype)
        print('tif rgb img shape', tif_RGB_img.shape, tif_RGB_img.dtype)
        print('kelp img shape', kelp_img.shape, kelp_img.dtype) 
        print('kelp rgb img shape', kelp_RGB_img.shape, kelp_RGB_img.dtype)
        
        print('/////////////////max tif =', np.max(tif_RGB_img))
        print('/////////////////max kelp =', np.max(kelp_RGB_img))
        
        # Blend the images using addWeighted
        alpha = 0.5 
        blended_img = cv2.addWeighted(tif_RGB_img, alpha, kelp_RGB_img, 1-alpha, 0)
        
        fig, axs = plt.subplots(1, 3, figsize=(3 * 5, 5))
        
        axs[0].imshow(tif_RGB_img, cmap='gray')  # Assuming grayscale for each channel
        axs[0].axis('off')
        axs[0].set_title('TIF Satellite RGB Image')
        
        axs[1].imshow(kelp_RGB_img, cmap='gray')  # Assuming grayscale for each channel
        axs[1].axis('off')
        axs[1].set_title('Kelp label Image')
        
        axs[2].imshow(blended_img, cmap='gray')  # Assuming grayscale for each channel
        axs[2].axis('off')
        axs[2].set_title('Blended Image')
        
        plt.show()
 
    except Exception as e:
        print("Error:", e)
        
# Provide the path to your .TIF image file
#tif_image_path = "C:/Users/thana/Downloads/test_satellite/GZ905340_satellite.tif"

#Show train pic and label
#tif_image_path = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/data/train_satellite/RZ775960_satellite.tif"
#kelp_image_path = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/data/train_kelp/RZ775960_kelp.tif"

#Show test results
tif_image_path = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/train_val_test_data/test_images/AA498489_satellite.tif"
kelp_image_path = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/train_val_test_data/test_labels/AA498489_kelp.tif"

kelp_image_path_1 = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/train_val_test_data/test_labels/CL954551_kelp.tif"
kelp_image_path_2 = "C:/Users/thana/Desktop/MSRobo_Work/4_spring2024/CS6476_CV/kelp_segmentation/results/CL954551_kelp.tif"

#show_tif_image(tif_image_path)
#show_kelp_image(kelp_image_path)
#show_7_ch_tif_image(tif_image_path)
#show_both_overlay(tif_image_path,kelp_image_path)
show_kelp_image(kelp_image_path_1)
show_kelp_image(kelp_image_path_2)

