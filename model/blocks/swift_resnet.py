import torch
import torchsummary
from torch import nn
import torchvision


class SwiftLaneResBlock(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int = 1):
        super(SwiftLaneResBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), (stride, stride), 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.ds = torch.nn.Identity()
        if stride != 1 or in_channel != out_channel:
            self.ds = torch.nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, 1), (stride, stride), 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.ac = nn.ReLU()

    def forward(self, x):
        xp = self.conv(x)
        xp = xp + self.ds(x)
        return self.ac(xp)

    @staticmethod
    def make_layers(in_channel: int, out_channel: int, layers: int, stride: int):
        strides = [stride] + [1] * (layers - 1)
        block = []
        for i in strides:
            block.append(SwiftLaneResBlock(in_channel, out_channel, i))
            in_channel = out_channel
        return nn.Sequential(*block)


class SwiftLaneResBottleneck(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int = 1):
        super(SwiftLaneResBottleneck, self).__init__()
        self.exp = 4
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, (3, 3), (stride, stride), 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * self.exp, (1, 1), (1, 1), 0, bias=False),
            nn.BatchNorm2d(out_channel * self.exp)
        )
        self.ds = torch.nn.Identity()
        if stride != 1 or in_channel != out_channel * self.exp :
            self.ds = torch.nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.exp, (1, 1), (stride, stride), 0, bias=False),
                nn.BatchNorm2d(out_channel * self.exp)
            )
        self.ac = nn.ReLU()

    def forward(self, x):
        xp = self.conv(x)
        xr = self.ds(x)
        x = xp + xr
        return self.ac(xp)

    @staticmethod
    def make_layers(in_channel: int, out_channel: int, layers: int, stride: int):
        strides = [stride] + [1] * (layers - 1)
        block = []
        for i in strides:
            block.append(SwiftLaneResBottleneck(in_channel, out_channel, i))
            in_channel = out_channel * 4
        return nn.Sequential(*block)


class SwiftLaneBaseResBlock(nn.Module):
    def __init__(self, channels, out_channels=64):
        super(SwiftLaneBaseResBlock, self).__init__()
        self.resnet_conv1 = nn.Conv2d(channels, out_channels, (7, 7), (2, 2), 3)
        self.resnet_bn1 = nn.BatchNorm2d(out_channels)
        self.resnet_relu1 = nn.ReLU()
        self.resnet_maxpool1 = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu1(x)
        x = self.resnet_maxpool1(x)
        return x

class SwiftLaneBaseResBlockAlter(nn.Module):
    def __init__(self, channels, out_channels=64):
        super(SwiftLaneBaseResBlockAlter, self).__init__()
        self.resnet_conv1 = nn.Conv2d(channels, out_channels, (7, 7), (2, 2), 3)
        self.resnet_bn1 = nn.BatchNorm2d(out_channels)
        self.resnet_relu1 = nn.ReLU()
        self.resnet_maxpool1 = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        y = self.resnet_conv1(x)
        
        x = self.resnet_bn1(y)
        x = self.resnet_relu1(x)
        x = self.resnet_maxpool1(x)
        return x, y



class ArvnResNet34(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(ArvnResNet34, self).__init__()
        self.base_block = SwiftLaneBaseResBlock(in_channels, base_channels)
        self.layer1 = SwiftLaneResBlock.make_layers(base_channels, base_channels, 3, 1)
        self.layer2 = SwiftLaneResBlock.make_layers(base_channels, base_channels * 2, 4, 2)
        self.layer3 = SwiftLaneResBlock.make_layers(base_channels * 2, base_channels * 4, 6, 2)
        self.layer4 = SwiftLaneResBlock.make_layers(base_channels * 4, base_channels * 8, 3, 2)

    def forward(self, x):
        x = self.base_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ArvnResNet18(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(ArvnResNet18, self).__init__()
        self.base_block = SwiftLaneBaseResBlock(in_channels, base_channels)
        self.layer1 = SwiftLaneResBlock.make_layers(base_channels, base_channels, 2, 1)
        self.layer2 = SwiftLaneResBlock.make_layers(base_channels, base_channels * 2, 2, 2)
        self.layer3 = SwiftLaneResBlock.make_layers(base_channels * 2, base_channels * 4, 2, 2)
        self.layer4 = SwiftLaneResBlock.make_layers(base_channels * 4, base_channels * 8, 2, 2)

    def forward(self, x):
        x = self.base_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ArvnResNet50(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(ArvnResNet50, self).__init__()
        self.base_block = SwiftLaneBaseResBlock(in_channels, base_channels)
        self.layer1 = SwiftLaneResBottleneck.make_layers(base_channels, base_channels, 3, 1)
        self.layer2 = SwiftLaneResBottleneck.make_layers(base_channels * 4, base_channels * 2, 4, 2)
        self.layer3 = SwiftLaneResBottleneck.make_layers(base_channels * 8, base_channels * 4, 6, 2)
        self.layer4 = SwiftLaneResBottleneck.make_layers(base_channels * 16, base_channels * 8, 3, 2)

    def forward(self, x):
        x = self.base_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ArvnResNetHelper:
    @staticmethod
    def get_arch(arch, in_channels=3, base_channels=64):
        if arch == "resnet18":
            return ArvnResNet18(in_channels, 64)
        elif arch == "resnet18_light":
            return ArvnResNet18(in_channels, 16)
        elif arch == "resnet34":
            return ArvnResNet34(in_channels, 64)
        elif arch == "resnet34_light":
            return ArvnResNet34(in_channels, 16)
        elif arch == "resnet50_o":
            return torchvision.models.resnet50(pretrained=True, progress=True)
        elif arch == "resnet50":
            return ArvnResNet50(in_channels, 64)
        elif arch == "resnet50_light":
            return ArvnResNet50(in_channels, 16)
        else:
            raise Exception("Model arch is not supported")


if __name__ == "__main__":
    model = ArvnResNet50(3).to("cpu")
    torchsummary.summary(model, (3, 512, 512), device="cpu")
