import torch.nn as nn
import torchvision.models as models


class TorchVisionResNet18(nn.Module):
    def __init__(self):
        super(TorchVisionResNet18, self).__init__()
        self.net = models.resnet18(weights=None)
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, 6)

    def forward(self, x):
        return self.net(x)