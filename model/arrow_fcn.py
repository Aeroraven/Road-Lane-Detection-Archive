import torchsummary
from torch import nn
import segmentation_models_pytorch as smp

from model.blocks.fcn_blocks import FCNSimpleDecoder, FCNConcatDecoder
from model.blocks.swift_resnet import ArvnResNetHelper, SwiftLaneBaseResBlock, SwiftLaneBaseResBlockAlter, SwiftLaneResBottleneck


class ArrowFCN(nn.Module):
    def __init__(self,
                 channels: int = 3):
        super(ArrowFCN, self).__init__()
        self.encoder = ArvnResNetHelper.get_arch("resnet50", channels)
        self.base_output = 2048
        self.decoder1 = FCNSimpleDecoder(self.base_output, self.base_output // 2)
        self.decoder2 = FCNSimpleDecoder(self.base_output // 2, self.base_output // 4)
        self.decoder3 = FCNSimpleDecoder(self.base_output // 4, self.base_output // 8)
        self.decoder4 = FCNSimpleDecoder(self.base_output // 8, self.base_output // 16)
        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.base_output // 16, self.base_output // 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(self.base_output // 32, self.base_output // 16, kernel_size=(3, 3), padding=1, bias=False)
        )
        self.output = nn.Conv2d(self.base_output // 16, 2, (1, 1))
        self.actv = nn.Softmax(1)
        # self.prt = smp.Unet(classes=2,activation="softmax")

    def forward(self, x):
        # return self.prt(x)
        x = self.encoder(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.output(x)
        x = self.actv(x)
        return x


class ArrowFCNEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(ArrowFCNEncoder, self).__init__()
        self.base_block = SwiftLaneBaseResBlockAlter(in_channels, base_channels)
        self.layer1 = SwiftLaneResBottleneck.make_layers(base_channels, base_channels, 3, 1)
        self.layer2 = SwiftLaneResBottleneck.make_layers(base_channels * 4, base_channels * 2, 4, 2)
        self.layer3 = SwiftLaneResBottleneck.make_layers(base_channels * 8, base_channels * 4, 6, 2)
        self.layer4 = SwiftLaneResBottleneck.make_layers(base_channels * 16, base_channels * 8, 3, 2)

    def forward(self, x):
        x, x0 = self.base_block(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x, x0, x1, x2, x3, x4


class ArrowFCNMK2(nn.Module):
    def __init__(self,
                 channels: int = 3):
        super(ArrowFCNMK2, self).__init__()
        self.encoder = ArrowFCNEncoder(channels)
        self.base_output = 2048
        self.decoder1 = FCNConcatDecoder(self.base_output, self.base_output, self.base_output // 2)
        self.decoder2 = FCNConcatDecoder(self.base_output // 2, self.base_output // 2, self.base_output // 4)
        self.decoder3 = FCNConcatDecoder(self.base_output // 4, self.base_output // 4, self.base_output // 8)
        self.decoder4 = FCNConcatDecoder(self.base_output // 8, self.base_output // 16 + self.base_output // 32, self.base_output // 16)
        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.base_output // 16, self.base_output // 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(self.base_output // 32, self.base_output // 16, kernel_size=(3, 3), padding=1, bias=False)
        )
        self.output = nn.Conv2d(self.base_output // 16, 2, (1, 1))
        self.actv = nn.Softmax(1)
        self.prt = smp.Unet("resnet50",encoder_weights=None,classes=2,activation="softmax")

    def forward(self, x):
        return self.prt(x)
        xf, x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder1(x4,x3)
        x = self.decoder2(x,x2)
        x = self.decoder3(x,x1)
        x = self.decoder4(x,x0)
        x = self.decoder5(x)
        x = self.output(x)
        x = self.actv(x)
        return x


if __name__ == "__main__":
    model = ArrowFCN(3).to("cpu")
    torchsummary.summary(model, (3, 480, 800), device="cpu")
