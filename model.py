import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, feature_dim, bias=False), nn.ReLU(inplace=True),
                               nn.Linear(feature_dim, feature_dim, bias=False))

    def forward(self, x):
        x = self.f(x)
        feature = F.normalize(torch.flatten(x, start_dim=1), dim=-1)
        out = F.normalize(self.g(feature), dim=-1)
        return feature, out
