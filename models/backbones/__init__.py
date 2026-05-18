from .resnet import resnet18, resnet34, resnet50


BACKBONE_REGISTRY = {
    "resnet18" : resnet18,
    "resnet34" : resnet34,
    "resnet50" : resnet50,
}