import torch
from torch.nn.modules import padding
from utils.WeightsInitializer import WeightsInitializer
from utils.densnets.DenseLayer import DenseLayer
from torch import nn


class DenseBlock(WeightsInitializer):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.conv_pass = nn.Sequential()
        for i in range(num_layers):
            self.conv_pass.append(DenseLayer(
                in_channels=in_channels+i*growth_rate,
                k=growth_rate
                ))

    def forward(self, X):
        out = X
        for conv_block in self.conv_pass:
            out = torch.concat([out, conv_block(out)], dim=1)
        return out



