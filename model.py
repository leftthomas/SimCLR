import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50


class Net(nn.Module):
    def __init__(self, model_type, features_dim=128):
        super(Net, self).__init__()
        if model_type == 'resnet18':
            self.features_extractor, expand = resnet18(), 1
        else:
            self.features_extractor, expand = resnet50(), 4
        self.features_extractor.fc = nn.Linear(512 * expand, features_dim)

    def forward(self, x):
        features = self.features_extractor(x)
        features = F.normalize(features)
        return features
