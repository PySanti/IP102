import torch
from torch import nn
from utils.WeightsInitializer import WeightsInitializer
from dropblock import DropBlock2D
from utils.resnets.ResBlock import ResBlock

class ResNet34(WeightsInitializer):
    def __init__(self, input_dim) -> None:
        super(ResNet34, self).__init__()
        self.initial_convolution = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.res_pass = nn.Sequential(
            ResBlock(input_channels=64, output_channels=64, downsampling=False),
            ResBlock(input_channels=64, output_channels=64, downsampling=False),
            ResBlock(input_channels=64, output_channels=64, downsampling=False),
            ResBlock(input_channels=64, output_channels=128, downsampling=True),
            DropBlock2D(0.3, block_size=15),
            ResBlock(input_channels=128, output_channels=128, downsampling=False),
            ResBlock(input_channels=128, output_channels=128, downsampling=False),
            ResBlock(input_channels=128, output_channels=128, downsampling=False),
            ResBlock(input_channels=128, output_channels=256, downsampling=True),
            DropBlock2D(0.3, block_size=8),
            ResBlock(input_channels=256, output_channels=256, downsampling=False),
            ResBlock(input_channels=256, output_channels=256, downsampling=False),
            ResBlock(input_channels=256, output_channels=256, downsampling=False),
            ResBlock(input_channels=256, output_channels=256, downsampling=False),
            ResBlock(input_channels=256, output_channels=256, downsampling=False),
            ResBlock(input_channels=256, output_channels=512, downsampling=True),
            DropBlock2D(0.3, block_size=4),
            ResBlock(input_channels=512, output_channels=512, downsampling=False),
            ResBlock(input_channels=512, output_channels=512, downsampling=False),
            )
        self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.linear_pass = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 102),
            )
        self._init_weights()
    def forward(self, X):
        out = self.initial_convolution(X)
        out = self.res_pass(out)
        out = self.global_pooling(out)
        return self.linear_pass(out)

