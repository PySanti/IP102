from utils.WeightsInitializer import WeightsInitializer
from torch import nn
from utils.densnets.DenseBlock import DenseBlock
from utils.densnets.DenseTransition import DenseTransition

class DenseNet121(WeightsInitializer):
    def __init__(self, in_channels, k=32, compression=0.5, dense_layers=[6, 12, 24, 16]):
        super(DenseNet121, self).__init__()
        self.initial_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        self.dense_pass = nn.Sequential()
        out = 64
        for i,dl in enumerate(dense_layers):
            self.dense_pass.append(DenseBlock(in_channels=out,growth_rate=k,num_layers=dl))
            if i != len(dense_layers)-1:
                self.dense_pass.append(DenseTransition(in_channels=int(out + dl*k), compression=compression))
                out = (out + dl*k)*compression

        self.linear_pass = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(int(out + dense_layers[-1]*k), 102),
                )
        self._init_weights()


    def forward(self, X):
        out = self.initial_conv(X)
        out = self.dense_pass(out)
        out = self.linear_pass(out)
        return out
