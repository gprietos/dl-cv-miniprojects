import torch
import torch.nn as nn

from models.base import BackboneOutput


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.fc = nn.Linear(in_features, num_classes)


    def forward(self, features: BackboneOutput):
        x = torch.flatten(features.pooled, 1)
        x = self.fc(x)

        return x
