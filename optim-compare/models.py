import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.non_linearity1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.non_linearity2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(9*9*32, num_classes)

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, output_size, stride=1, down_sampling=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.conv2 = nn.Conv2d(output_size, output_size,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, self.expansion * output_size,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * output_size)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_size, output_size, stride=1, down_sampling=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.conv2 = nn.Conv2d(
            output_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.conv3 = nn.Conv2d(
            output_size, output_size * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_size * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sampling = down_sampling
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sampling is not None:
            identity = self.down_sampling(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.input_size = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, output_size, blocks, stride=1):
        down_sampling = None
        if stride != 1 or self.input_size != output_size * block.expansion:
            down_sampling = nn.Sequential(
                nn.Conv2d(self.input_size, output_size * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_size * block.expansion),
            )

        layers = []
        layers.append(
            block(self.input_size, output_size, stride, down_sampling))
        self.input_size = output_size * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.input_size, output_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ResNet18
# model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=number of classes you want to classify)

# ResNet34
# model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=number of classes you want to classify)

# ResNet50
# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=number of classes you want to classify)

# ResNet101
# model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=number of classes you want to classify)

# ResNet150
# model = ResNet(Bottleneck, [3, 6, 36, 3], num_classes=number of classes you want to classify)


class MLPConvNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MLPConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(8*8*32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        # print(x.shape) # torch.Size([100, 1, 28, 28])
        out = self.layer1(x)
        # print(out.shape) # torch.Size([100, 16, 15, 15])
        out = self.layer2(out)
        # print(out.shape) # torch.Size([100, 32, 8, 8])
        out = out.reshape(out.size(0), -1)
        #x = x.view(-1, 28 * 28)
        # print(out.shape)
        out = self.mlp(out)  # + self.mlp(x)
        return out
