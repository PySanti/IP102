import torch
from torch import nn
from torch.nn.modules import BatchNorm2d


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, downsampling):
        super(ResBlock, self).__init__()
        self.downsampling = None if not downsampling else nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels)
            )
        self.res_pass = nn.Sequential(
                nn.Conv2d(
                            input_channels,
                            output_channels, 
                            kernel_size=3, 
                            stride=2 if self.downsampling else 1,
                            padding = 1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(
                            output_channels,
                            output_channels, 
                            kernel_size=3, 
                            stride=1,
                            padding = 1),
                nn.BatchNorm2d(output_channels)
                )
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.res_pass(X)
        if self.downsampling:
            X = self.downsampling(X)
        X = X+out
        return self.relu(X)

    def _init_weights(self):
        pass
