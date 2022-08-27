import torch
from torch import nn


class FCNSimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNSimpleDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.conv_relu(x1)
        return x1


class FCNConcatDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(FCNConcatDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1
