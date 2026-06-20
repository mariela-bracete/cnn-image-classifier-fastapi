import torch.nn as nn
import torch.nn.functional as F

class AssignmentCNN(nn.Module):
    """
    CNN architecture matching Assignment 2 specifications.

    Input: 64x64x3
    Conv2D(16, 3x3, stride=1, padding=1)
    ReLU
    MaxPool2D(2x2)

    Conv2D(32, 3x3, stride=1, padding=1)
    ReLU
    MaxPool2D(2x2)

    Flatten
    FC(100)
    ReLU
    FC(10)
    """

    def __init__(self):
        super(AssignmentCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64x64
        # -> conv1 = 64x64x16
        # -> pool = 32x32x16
        # -> conv2 = 32x32x32
        # -> pool = 16x16x32

        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
class EnhancedCNN(nn.Module):
    """
    CNN architecture based on the Module 3 Practical 3 CNN notebook.
    Designed for CIFAR-10 images with shape 3 x 32 x 32.
    """

    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 128 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x