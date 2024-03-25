from math import floor

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import Dataset, random_split, DataLoader, dataloader
import os
from tifffile import imread
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from cnn_class import cnn_architecture


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = None
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        lst = os.listdir(self.img_dir)
        number_img = len(lst)
        return number_img

    def __getitem__(self, idx):
        # read img from folder in order of idx using os.listdir
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        # read image
        image = imread(img_path)
        img_array = np.array(image)
        # get the rgb image
        rgb_img, _ = np.split(img_array, [3], axis=2)
        rgb_img = rgb_img.transpose(2, 0, 1)
        # print('rgb_img shape:', rgb_img.shape)
        image = torch.tensor(rgb_img) / 255.0

        # get the label
        # get the NDVI (Normalized Difference Vegetation Index)
        """
        1. SWIR (Shortwave Infrared)
        2. NIR (Near-Infrared)
        3. Red
        4. Green
        5. Blue
        6. Cloud Mask
        7. Digital Elevation Model
        """
        Red = img_array[:, :, 2]
        NIR = img_array[:, :, 1]
        temp1 = Red - NIR
        temp2 = Red + NIR
        NDVI = temp1 / temp2
        if np.mean(NDVI) > 0.5:
            label = 1
        else:
            label = 0

        label = float(label)
        # print('label:', label)
        # print('image:', image.shape)

        return image.float(), label



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = cnn_architecture().to(device)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Perform training loop for n epochs
loss_list = []
n_epochs = 10
loss_fn = nn.MSELoss()


def train_model(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.

    for batch, (images, targets) in enumerate(dataloader):
        # print(targets.shape)

        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        output = output.squeeze()
        # print(output.shape)
        batch_loss = loss_fn(output.float(), targets.float())
        batch_loss.backward()
        optimizer.step()

        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(images)
        epoch_loss = (epoch_loss * batch + batch_loss) / (batch + 1)
        print(f"loss: {batch_loss:>7f} [{sample_count:>5d}/{size:>5d}]")

    return epoch_loss


def test_model(model, dataloader, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            pred = pred.squeeze()
            loss = loss_fn(pred, targets)
            test_loss += loss.item()

    test_loss /= num_batches

    return test_loss


# Load the datset (split into train and test)
collected_data = CustomImageDataset(img_dir='/Users/nadira/gatech/Sp24/CV/kelp_segmentation/data/train_satellite')
# Start training
train_data_len = len(collected_data)
train_data_size = floor(train_data_len * 0.9)
test_data_size = round(train_data_len * 0.1)

train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
train_dataloader = DataLoader(train_data, batch_size=27)
test_dataloader = DataLoader(test_data, batch_size=27)
epochs = 15

for t in range(n_epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_model(model, train_dataloader, loss_fn, optimizer)
    loss = test_model(model, test_dataloader, loss_fn)
    loss_list.append(loss)
    print(f"\nTest Error: \n----------\n{loss:.6f}")

# Plot the loss
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()

# Save model
save_path = 'first_model.pth'
torch.save(model.state_dict(), save_path)
