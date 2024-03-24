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

model.save("model_CNN_v1.h5")