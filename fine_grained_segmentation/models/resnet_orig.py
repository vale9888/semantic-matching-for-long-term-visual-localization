# Script originally from https://github.com/maunzzz/fine-grained-segmentation-networks,
# licensed as in the LICENSE file of the above repository (Attribution-NonCommercial 4.0 International).

import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1, dilation=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    ## Not sure why we call it bottleneck but it is a residual module, where we apply first a conv 1x1, then a 3x3 (and depth is as specified ì) and finally we sum the original input.
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride=stride, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, stride=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)        

        if self.downsample is not None:
            identity = self.downsample(x)
        ## Here we sum the original input
        out += identity
        out = self.relu(out)

        return out

class ResNetForPsp(nn.Module):

    ## requires a few more parameters in initialization
    def __init__(self, block, layers):
        super(ResNetForPsp, self).__init__()
        self.inplanes = 64
        ## 3 in channels cause it's RGB
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## layers=n_blocks=[3, 4, 23, 3]... I am guessing this has to be tuned in a regular ResNet
        ## so far it's H/2xW/2x64
        self.layer1 = self._make_layer(block, 64, layers[0])
        ## so far it's H/2xW/2x256, inplanes 256... then I'm lost
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: ## block.expansion == 4 (see Bottleneck class) - class is not yet init but the expansion parameter is there anyway
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ## Here we do init Bottleneck - we append one such module for n_blocks=[3, 4, 23, 3] times respectively
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        h1 = self.layer3(x)
        h2 = self.layer4(h1)

        ## probably because we are using the pyramid layer we need both the last and second-last layer
        if self.training:
            return h1, h2
        else:
            return h2


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetForPsp(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    return model



