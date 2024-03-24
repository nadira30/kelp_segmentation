# kelp_segmentation
Computer Vision Class project: https://www.drivendata.org/competitions/255/kelp-forest-segmentation/data/



# Boss commit

test_TIF_import_plot.py: Test importing the TIF file, select channel(s), then plot them

import_data_split.py: (active) Import TIF images/labels in train folders, then use the metadata.csv to split train and val, then normalize, and save the preprocessed data in .npy form to be processed before training in CNN. 

import_data_split_2.py: (inactive) same as import_data_split.py but add additional process of deducting channel 0 (short-wave IR) from channel 1 (near IR)

CNN_model.py: (active) import the preprocessed npy, then put in the base CNN model to train, when finished, save the model and weights as h5 file
CNN_model_2.py: (inactive) same as CNN_model.py but support the data from import_data_split_2.py

predict_and_save_img.py: (active) load CNN model and weights file, load the test img, then predict, post process to make sure that label image contains only 0 and 1, then save in the "results" folder 