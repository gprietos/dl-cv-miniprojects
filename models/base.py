from dataclasses import dataclass, field
import torch
from typing import List, Optional
import torch.nn as nn


@dataclass
class BackboneOutput:
    """
    The shared contract every backbone must return.
    Necks consume and return this same type.
    Heads receive this and pick only what they need.
    """

    feature_map: torch.Tensor
    pooled: torch.Tensor
    stages: List[torch.Tensor]



class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)

        return out
