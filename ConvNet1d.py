import torch
import torch.nn as nn


# Convolutional neural network (two convolutional layers)
class ConvNet1d(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=80, stride=80),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4))
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4))
        self.avgpool = nn.AvgPool1d(kernel_size = 500)
        """
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=(2,2)),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=(2,2)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        """
        self.fc = nn.Sequential(
            nn.Softmax(dim=1))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avgpool(out)
        #out = out.reshape(out.size(0, -1))
        out = self.fc(out)
        return out