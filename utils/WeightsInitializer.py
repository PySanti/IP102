from torch import nn
from torch.nn import init

class WeightsInitializer(nn.Module):
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Inicializa los pesos de las capas convolucionales
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Inicializa los sesgos a cero
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Inicializa los pesos de batch norm a 1 y sesgos a 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Inicializa las capas lineales
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

