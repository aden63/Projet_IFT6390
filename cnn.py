import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ConvNet(nn.Module):
    def __init__(self, kernelNumber1, kernelNumber2, kernelSize1, kernelSize2, num_classes=10):
        super(ConvNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, kernelNumber1, kernel_size=kernelSize1, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        outputSize1 = (28 - kernelSize1 + 2 * 2 + 1) / 2

        # Layer 2
        self.conv2 = nn.Conv2d(kernelNumber1, kernelNumber2, kernel_size=kernelSize2, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        outputSize2 = (outputSize1 - kernelSize2 + 2 * 2 + 1) / 2

        # Initialization
        self.layer1 = nn.Sequential(
            self.conv1,
            self.batchNorm1,
            self.relu1,
            self.maxPool1)

        self.layer2 = nn.Sequential(
            self.conv2,
            self.batchNorm2,
            self.relu2,
            self.maxPool2)
        self.fc = nn.Linear(int(outputSize2 ** 2 * kernelNumber2), num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out