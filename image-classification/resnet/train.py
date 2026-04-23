import wandb
import albumentations as A
import torch
import torch.nn as nn


model = torch.nn.Sequaential(
    torch.nn.Conv2d()
)

torch.nn.init.kaiming_normal_


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)