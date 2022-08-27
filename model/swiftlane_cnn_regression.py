import torch
from torch import nn

from model.blocks.swift_resnet import SwiftLaneResBlock


class SwiftLaneRegressionCNN(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 height: int,
                 width: int,
                 c: int,
                 h: int,
                 w: int):
        super(SwiftLaneRegressionCNN, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.ch = c
        self.hl = h
        self.wd = w

        # Residual CNN Blocks
        # Initial Block
        self.resnet_conv1 = nn.Conv2d(channels, 64, (7, 7), (2, 2), 3)
        self.resnet_bn1 = nn.BatchNorm2d(64)
        self.resnet_relu1 = nn.ReLU()
        self.resnet_maxpool1 = nn.MaxPool2d(3, 2, 1)

        # First Res Block
        self.resnet_layer1 = SwiftLaneResBlock.make_layers(64, 64, 2, 1)

        # Second Res Block
        self.resnet_layer2 = SwiftLaneResBlock.make_layers(64, 128, 2, 2)

        # Third Res Block
        self.resnet_layer3 = SwiftLaneResBlock.make_layers(128, 256, 2, 2)

        # SwiftLane: Pooling & Conv
        self.swift_maxpool1 = nn.MaxPool2d(2)
        self.swift_conv1 = nn.Conv2d(256, 8, (1, 1))

        # Estimator
        wp = torch.ones((1, 3, height, width))
        self.feature_shape = self.conv_forwarding(wp).shape

        # SwiftLane: Fully Connected
        self.swift_fl = nn.Flatten()
        self.swift_drop1 = nn.Dropout(0.2)
        self.swift_fc2 = nn.Linear(self.feature_shape[1] * self.feature_shape[2] * self.feature_shape[3], 2048)
        self.swift_drop2 = nn.Dropout(0.2)
        self.swift_fc3 = nn.Linear(2048,512)
        self.swift_out = nn.Linear(512, c*3+4)
        self.swift_out2 = nn.Linear(512,c*2)
        self.swift_outfl = nn.Flatten()
        # Activation
        self.swift_actv = nn.Softmax(dim=-1)

    def conv_forwarding(self, x: torch.Tensor):
        # ResNet: Initial Conv
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_maxpool1(x)
        # ResNet: Block 1
        x = self.resnet_layer1(x)
        # ResNet: Block 2
        x = self.resnet_layer2(x)
        # ResNet: Block 3
        x = self.resnet_layer3(x)
        # SwiftLane: Conv
        x = self.swift_maxpool1(x)
        x = self.swift_conv1(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.conv_forwarding(x)
        x = self.swift_fl(x)
        x = self.swift_drop1(x)
        x = self.swift_fc2(x)
        x = self.swift_drop2(x)
        x = self.swift_fc3(x)
        x_params = self.swift_out(x)
        x_exist = self.swift_out2(x)
        x_exist = torch.reshape(x_exist,(x_exist.shape[0],self.ch,2))
        x_exist = self.swift_actv(x_exist)
        x_exist = torch.reshape(x_exist, (x_exist.shape[0], self.ch*2))
        x = torch.cat((x_params,x_exist),-1)
        return x