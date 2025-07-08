import torch.nn as nn
import torch.nn.functional as F

class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 28x28 -> 26x26
        self.pool = nn.MaxPool2d(2, 2)                # -> 13x13
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # -> 11x11
        self.pool2 = nn.MaxPool2d(2, 2)               # -> 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 47)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 13, 13)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 64, 5, 5)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
