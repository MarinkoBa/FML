import torch
import torch.nn as nn
import numpy as np


class Q_Net(nn.Module):

    def __init__(self):
        super(Q_Net, self).__init__()

        self.conv1 = ConvNormRelu(5, 32, kernel_size=3, stride=2)
        self.conv2 = ConvNormRelu(32, 32, kernel_size=3, stride=2)
        self.conv3 = ConvNormRelu(32, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(32, 6)

    def forward(self, x):
        x = x.to(device='cuda:0')

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x.reshape(x.shape[0], 32))

        return x

class ConvNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvNormRelu, self).__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convolution(x)
        #x = self.batch_norm(x)
        x = self.relu(x)
        return x
