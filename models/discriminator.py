# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

"""
Implementation of Discriminator (https://arxiv.org/pdf/1802.07934.pdf)
"""

from torch import nn


class disc_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, leaky=None):
        super(disc_conv2d, self).__init__()

        self.leaky = leaky

        if self.leaky:
            self.leakyReLU = nn.LeakyReLU(leaky, inplace=True)

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        x = self.conv(x)

        if self.leaky:
            x = self.leakyReLU(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.conv1 = disc_conv2d(num_classes, 64, leaky=0.2)
        self.conv2 = disc_conv2d(64, 128, leaky=0.2)
        self.conv3 = disc_conv2d(128, 256, leaky=0.2)
        self.conv4 = disc_conv2d(256, 512, leaky=0.2)
        self.conv5 = disc_conv2d(512, 1)
        self.activation = nn.Sigmoid()
        self.num_classes = num_classes

    def forward(self, x, size):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.activation(x)

        return x
