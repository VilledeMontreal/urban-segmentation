# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

"""
Implementation of Fast SCNN (https://arxiv.org/pdf/1902.04502.pdf)
"""

import torch
from torch import nn
from torch.nn import functional as F


class Fast_SCNN(torch.nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Fast_SCNN, self).__init__()

        self.num_classes = num_classes
        self.learningDownsample = LearningDownsample(input_channel)
        self.globalFeatureExtractor = GlobalFeatureExtractor()
        self.featureFusion = FeatureFusionModule()
        self.classifier = Classifier(128, num_classes)
        self.loss_fct = nn.BCELoss()       # Not used in semi-supervised training

    def forward(self, x, *args):
        downsampled = self.learningDownsample(x)
        x = self.globalFeatureExtractor(downsampled)
        x = self.featureFusion(x, downsampled)
        x = self.classifier(x)

        return x

    def loss(self, x, y):
        # Not used in semi-supervised training
        loss = self.loss_fct(x, y)

        return loss


class LearningDownsample(torch.nn.Module):
    def __init__(self, input_channel):
        super().__init__()

        channel_depth_0 = 32
        channel_depth_1 = 48
        channel_depth_2 = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=channel_depth_0,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_0),
            nn.ReLU())

        self.dsconv1 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_0,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      groups=channel_depth_0,
                      bias=False),
            nn.BatchNorm2d(channel_depth_0),
            # Pointwise
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_1,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_1),
            nn.ReLU(inplace=True))

        self.dsconv2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=channel_depth_1,
                      out_channels=channel_depth_1,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      groups=channel_depth_1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_1),
            # Pointwise
            nn.Conv2d(in_channels=channel_depth_1,
                      out_channels=channel_depth_2,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)

        return x


class GlobalFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channel_depth_0 = 64
        channel_depth_1 = 64
        channel_depth_2 = 96
        channel_depth_3 = 128
        channel_depth_p = 128

        self.bottleneck1 = nn.Sequential(
            Bottleneck(input_channel=channel_depth_0,
                       output_channel=channel_depth_1,
                       stride=2),
            Bottleneck(input_channel=channel_depth_1,
                       output_channel=channel_depth_1,
                       stride=1,
                       residual=True),
            Bottleneck(input_channel=channel_depth_1,
                       output_channel=channel_depth_1,
                       stride=1,
                       residual=True))

        self.bottleneck2 = nn.Sequential(
            Bottleneck(input_channel=channel_depth_1,
                       output_channel=channel_depth_2,
                       stride=2),
            Bottleneck(input_channel=channel_depth_2,
                       output_channel=channel_depth_2,
                       stride=1,
                       residual=True),
            Bottleneck(input_channel=channel_depth_2,
                       output_channel=channel_depth_2,
                       stride=1,
                       residual=True))

        self.bottleneck3 = nn.Sequential(
            Bottleneck(input_channel=channel_depth_2,
                       output_channel=channel_depth_3,
                       stride=1),
            Bottleneck(input_channel=channel_depth_3,
                       output_channel=channel_depth_3,
                       stride=1,
                       residual=True),
            Bottleneck(input_channel=channel_depth_3,
                       output_channel=channel_depth_3,
                       stride=1,
                       residual=True))

        self.ppm = PyramidPooling(input_channel=channel_depth_3,
                                  output_channel=channel_depth_p,
                                  sizes=(1, 2, 3, 6))

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)

        return x


# Based on MobileNetv2: https://arxiv.org/pdf/1801.04381.pdf
class Bottleneck(torch.nn.Module):
    def __init__(self, input_channel, output_channel, stride,
                 residual=False, expansion_factor=6):
        super(Bottleneck, self).__init__()
        self.residual = residual

        expand_channel = input_channel * expansion_factor

        self.dsconv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels=input_channel,
                      out_channels=expand_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(expand_channel),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(in_channels=expand_channel,
                      out_channels=expand_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=expand_channel,
                      bias=False),
            nn.BatchNorm2d(expand_channel),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels=expand_channel,
                      out_channels=output_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(output_channel))

    def forward(self, x):
        if self.residual:
            return x + self.dsconv(x)
        else:
            return self.dsconv(x)


class PyramidPooling(nn.Module):
    def __init__(self, input_channel, output_channel, sizes):
        super().__init__()

        self.layers = []
        for size in sizes:
            self.layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels=input_channel,
                          out_channels=input_channel,
                          kernel_size=1,
                          bias=False)))

        self.layers = nn.ModuleList(self.layers)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=(input_channel * (len(sizes) + 1)),
                      out_channels=output_channel,
                      kernel_size=1),
            nn.ReLU())

    def forward(self, x):
        input_size = x.size()
        out = []

        for layer in self.layers:
            out.append(F.interpolate(
                input=layer(x),
                size=(input_size[2], input_size[3]),
                mode='bilinear',
                align_corners=True))

        out.append(x)
        out = torch.cat(out, 1)

        out = self.conv(out)

        return out


class FeatureFusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channel_depth_lr = 128
        channel_depth_hr = 64
        channel_depth_out = 128

        self.dwconv_lr = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_lr,
                      out_channels=channel_depth_lr,
                      kernel_size=3,
                      stride=1,
                      padding=4,
                      dilation=4,
                      groups=128,
                      bias=False),
            nn.BatchNorm2d(channel_depth_lr),
            nn.ReLU())

        self.conv_lr = nn.Conv2d(in_channels=channel_depth_lr,
                                 out_channels=channel_depth_out,
                                 kernel_size=1)

        self.conv_hr = nn.Conv2d(in_channels=channel_depth_hr,
                                 out_channels=channel_depth_out,
                                 kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, lr_input, hr_input):
        lr_input = F.interpolate(input=lr_input,
                                 size=tuple(hr_input.shape[2:4]),
                                 mode='bilinear',
                                 align_corners=True)
        lr_input = self.dwconv_lr(lr_input)
        lr_input = self.conv_lr(lr_input)

        hr_input = self.conv_hr(hr_input)

        x = torch.add(hr_input, lr_input)
        x = self.relu(x)

        return x


class Classifier(torch.nn.Module):
    def __init__(self, input_channel, num_class):
        super().__init__()

        channel_depth_0 = 128
        channel_depth_1 = 128

        self.dsconv1 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=input_channel,
                      out_channels=input_channel,
                      kernel_size=3,
                      padding=1,
                      groups=input_channel,
                      bias=False),
            nn.BatchNorm2d(input_channel),
            # Pointwise
            nn.Conv2d(in_channels=input_channel,
                      out_channels=channel_depth_0,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_0),
            nn.ReLU(inplace=True))

        self.dsconv2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_0,
                      kernel_size=3,
                      padding=1,
                      groups=channel_depth_0,
                      bias=False),
            nn.BatchNorm2d(channel_depth_0),
            # Pointwise
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_1,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth_1),
            nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(in_channels=channel_depth_1,
                              out_channels=num_class,
                              kernel_size=1,
                              bias=True)

        self.upsample = nn.modules.UpsamplingBilinear2d(scale_factor=8)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.softmax(x)

        return x
