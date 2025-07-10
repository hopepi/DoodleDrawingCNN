import torch
import torch.nn as nn
import torch.nn.functional as F

class DoodleCNN(nn.Module):
    def __init__(self, num_classes=340):
        super(DoodleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2), # 5x5 kernel, padding=2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 3x3 kernel, padding=1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 3x3 kernel, padding=1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 3x3 kernel, padding=1
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 3x3 kernel, padding=1
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(512 * 2 * 2, 1024)  # feature map size 2x2, 512 kanal
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # Input: (batch,1,64,64) → Output: (batch,64,32,32)
        x = self.conv2(x)  # → (batch,128,16,16)
        x = self.conv3(x)  # → (batch,256,8,8)
        x = self.conv4(x)  # → (batch,256,4,4)
        x = self.conv5(x)  # → (batch,512,2,2)
        x = self.conv6(x)  # → (batch,512,2,2)

        x = x.view(x.size(0), -1)  # Flatten: (batch, 512*2*2)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
