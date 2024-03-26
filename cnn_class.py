import torch
import torch.nn as nn

class cnn_architecture(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*37*37, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2*350*350)
        self.relu = nn.ReLU()

    def forward(self, x):               #350x350
        x = self.relu(self.conv24(x))  # (350-5)/2+1 = 173
        x = self.relu(self.conv32(x))  # (173-5)/2+1 = 85
        x = self.relu(self.conv64_5(x))  # (85-5)/2+1 = 41
        x = self.relu(self.conv64_3(x))  # 41-3+1 = 39
        x = self.relu(self.conv64_3(x))  # 39-3+1 = 37

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x