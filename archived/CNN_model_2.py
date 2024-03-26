import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


preprocessed_data_dir = "./preprocessed_data/"

# Load preprocessed data
full_train_images = np.load(os.path.join(preprocessed_data_dir, "full_train_images.npy"))
full_train_labels = np.load(os.path.join(preprocessed_data_dir, "full_train_labels.npy"))
full_val_images = np.load(os.path.join(preprocessed_data_dir, "full_val_images.npy"))
full_val_labels = np.load(os.path.join(preprocessed_data_dir, "full_val_labels.npy"))

full_train_images = np.expand_dims(full_train_images, axis=-1)
full_train_labels = np.expand_dims(full_train_labels, axis=-1)
full_val_images = np.expand_dims(full_val_images, axis=-1)
full_val_labels = np.expand_dims(full_val_labels, axis=-1)


full_train_labels_resized = full_train_labels[:, 1:-1, 1:-1, :]
full_val_labels_resized = full_val_labels[:, 1:-1, 1:-1, :]


print("Size check - imported into NN")
print("full train imgs and resized lebels shape", full_train_images.shape, full_train_labels_resized.shape)
print("full val imgs and resized lebels shape", full_val_images.shape, full_val_labels_resized.shape)

print("------------full train labels unique number check-------------",  np.unique(full_train_labels_resized))
print("------------full val labels unique number check-------------",  np.unique(full_val_labels_resized))



image_height = full_train_images.shape[1]
image_width = full_train_images.shape[2]
num_channels = full_train_images.shape[3]


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_height, image_width,num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

print(model.summary())

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train
history = model.fit(full_train_images, full_train_labels_resized, epochs=10, batch_size=32, validation_data=(full_val_images, full_val_labels_resized))

# Evaluate the model
#loss, accuracy = model.evaluate(full_val_images, full_val_labels_resized)
#print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

model.save('CNN_model_v3.h5')
model.save_weights('CNN_model_weights_v3.h5')


