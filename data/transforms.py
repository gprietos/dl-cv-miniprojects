import torch
from torchvision.transforms import v2 as transforms


# ImageNet normalization constants
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def get_classification_transforms(split: str, img_size: int=224, backbone: str="resnet"):

    if split is not "train":
        return transforms.Compose([
            transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ])

    if backbone == "vit":
        return transforms.Compose([    
            transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),  
            transforms.ToImage(),
            transforms.ToDtype(dtype=torch.float32, scale=True),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            transforms.RandomErasing(p=0.25, value="random")
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
    ])