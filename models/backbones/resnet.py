import torch.nn as nn

from ..base import BackboneOutput

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # bias=False (BatchNorm will handle the bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.relu = nn.ReLU(inplace=True)                                                                       # inplace=True means that it will modify the input directly, without allocating any additional output.
        self.downsample = downsample                                                                            # None or a nn.Sequential with 1x1 conv + BN

    def forward(self, x):
        
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        return self.relu(x)



class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        mid_channels = out_channels // self.expansion

        super().__init__()

        # 1x1: compress
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3: spatial
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1: expand
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity

        return self.relu(x)


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super().__init__()

        self.in_channels = 64                                                                                   # tracks channels as we build layers
        
        # stem
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully conv body
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Pooling 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_features = 512 * block.expansion


    def _make_layer(self, block, out_channels, num_blocks, stride):
        # This helper builds one group of num_blocks BasicBlocks.

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        else:
            downsample = None

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)        


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_stages(self, stages):
        for stage in stages:
            module = getattr(self, stage)

            for p in module.parameters():
                p.requires_grad = False


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        hs1 = self.layer1(x)
        hs2 = self.layer2(hs1)
        hs3 = self.layer3(hs2)
        hs4 = self.layer4(hs3)

        pooled = self.avgpool(hs4)

        return BackboneOutput(
            feature_map=hs4,
            pooled=pooled,
            stages=[hs1, hs2, hs3, hs4]
        )


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(BottleneckBlock, [3,4,6,3])