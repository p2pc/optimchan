import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.non_linearity1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.non_linearity2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(8*8*32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pool1(x)
        x = self.non_linearity1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool2(x)
        x = self.non_linearity2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
