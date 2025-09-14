from utils.WeightsInitializer import WeightsInitializer
from torch import nn

class DenseTransition(WeightsInitializer):
    def __init__(self, in_channels, compression):
        super(DenseTransition, self).__init__()

        self.trans_pass = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=int(in_channels*compression), 
                    stride=1, 
                    padding=0, 
                    kernel_size=1
                ),
                nn.AvgPool2d(kernel_size=2, stride=2)
            );
        self._init_weights()

    def forward(self, X):
        return self.trans_pass(X)

