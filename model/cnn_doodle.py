import torch
import torch.nn as nn
import torch.nn.functional as F

class DoodleCNN(nn.Module):
    def __init__(self, num_classes=340):
        super(DoodleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.35)

        """
        Son Conv katmanın çıktısı 256 kanal var,
        Her kanalın boyutu 4x4 (yükseklik x genişlik),
        Toplam giriş boyutu = 256 * 4 * 4 = 4096 (flatten edilmiş),
        Bu fully connected katman 4096 boyutundaki girdi vektörünü
        512 boyutlu bir vektöre dönüştürüyor özellik çıkarımı için
        """
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
