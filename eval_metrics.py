import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import jaccard_score, roc_curve
#from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc

# Function to calculate Dice coefficient
# def dice_coefficient(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     union = np.sum(y_true) + np.sum(y_pred)
#     dice = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
#     return dice

# Calculate Dice Coefficient
def dice_coefficient(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places


# Load predicted and ground truth labels
results_dir = "./results"
test_labels_dir = "./train_val_test_data/test_labels"

predicted_images = []
ground_truth_images = []

for file in os.listdir(results_dir):
    if file.endswith(".tif"):
        predicted_image = cv2.imread(os.path.join(results_dir, file), cv2.IMREAD_GRAYSCALE)
        predicted_images.append(predicted_image)

for file in os.listdir(test_labels_dir):
    if file.endswith(".tif"):
        ground_truth_image = cv2.imread(os.path.join(test_labels_dir, file), cv2.IMREAD_GRAYSCALE)
        ground_truth_images.append(ground_truth_image)

# Convert lists to numpy arrays
predicted_images = np.array(predicted_images)
ground_truth_images = np.array(ground_truth_images)

# Flatten the images
predicted_images_flat = predicted_images.flatten()
ground_truth_images_flat = ground_truth_images.flatten()
print(np.sum(predicted_images_flat))
print(np.sum(ground_truth_images_flat))

# Binarize the images
predicted_images_bin = (predicted_images_flat > 0).astype(int)
ground_truth_images_bin = (ground_truth_images_flat > 0).astype(int)

# Calculate evaluation metrics
dice = dice_coefficient(ground_truth_images_bin, predicted_images_bin)
accuracy = accuracy_score(ground_truth_images_bin, predicted_images_bin)
precision = precision_score(ground_truth_images_bin, predicted_images_bin)
recall = recall_score(ground_truth_images_bin, predicted_images_bin)
f1 = f1_score(ground_truth_images_bin, predicted_images_bin)
iou = jaccard_score(ground_truth_images_bin, predicted_images_bin)
auc_roc = roc_auc_score(ground_truth_images_bin, predicted_images_bin)

# Print the evaluation metrics
print("Dice Coefficient:", dice)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("IoU:", iou)
print("Area under ROC curve:", auc_roc)
