from torch import nn
from utils.WeightsInitializer import WeightsInitializer

class DenseLayer(WeightsInitializer):
    def __init__(self, in_channels, k):
        super(DenseLayer, self).__init__()
        self.conv_pass = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4*k,
                kernel_size=1,
                padding=0,
                stride=1),
            nn.BatchNorm2d(4*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=4*k,
                out_channels=k,
                kernel_size=3,
                padding=1,
                stride=1),
            )
        self._init_weights()
    def forward(self, X):
        return self.conv_pass(X)
