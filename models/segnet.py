# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

"""
Implementation of SegNet (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)
Using pre-trained VGG16 as encoder
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SegNet(nn.Module):
    def __init__(self, input_channel, num_classes, pretrained=True):
        super(SegNet, self).__init__()

        self.input_channel = input_channel
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.loss_fct = nn.BCELoss()

        # VGG Architecture
        channel_depth_0 = 64
        channel_depth_1 = 128
        channel_depth_2 = 256
        channel_depth_3 = 512
        channel_depth_4 = 512

        # Encoder Hidden Layer 0
        self.enc_conv_00 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel,
                      out_channels=channel_depth_0,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_0))

        self.enc_conv_01 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_0,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_0))

        # Encoder Hidden Layer 1
        self.enc_conv_10 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_0,
                      out_channels=channel_depth_1,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_1))

        self.enc_conv_11 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_1,
                      out_channels=channel_depth_1,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_1))

        # Encoder Hidden Layer 2
        self.enc_conv_20 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_1,
                      out_channels=channel_depth_2,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_2))

        self.enc_conv_21 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_2,
                      out_channels=channel_depth_2,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_2))

        self.enc_conv_22 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_2,
                      out_channels=channel_depth_2,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_2))

        # Encoder Hidden Layer 3
        self.enc_conv_30 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_2,
                      out_channels=channel_depth_3,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_3))

        self.enc_conv_31 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_3,
                      out_channels=channel_depth_3,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_3))

        self.enc_conv_32 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_3,
                      out_channels=channel_depth_3,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_3))

        # Encoder Hidden Layer 4
        self.enc_conv_40 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_3,
                      out_channels=channel_depth_4,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_4))

        self.enc_conv_41 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_4,
                      out_channels=channel_depth_4,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_4))

        self.enc_conv_42 = nn.Sequential(
            nn.Conv2d(in_channels=channel_depth_4,
                      out_channels=channel_depth_4,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channel_depth_4))

        # Decoder Hidden Layer 4
        self.dec_conv_42 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_4,
                               out_channels=channel_depth_4,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_4))

        self.dec_conv_41 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_4,
                               out_channels=channel_depth_4,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_4))

        self.dec_conv_40 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_4,
                               out_channels=channel_depth_3,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_3))

        # Decoder Hidden Layer 3
        self.dec_conv_32 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_3,
                               out_channels=channel_depth_3,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_3))

        self.dec_conv_31 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_3,
                               out_channels=channel_depth_3,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_3))

        self.dec_conv_30 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_3,
                               out_channels=channel_depth_2,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_2))

        # Decoder Hidden Layer 2
        self.dec_conv_22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_2,
                               out_channels=channel_depth_2,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_2))

        self.dec_conv_21 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_2,
                               out_channels=channel_depth_2,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_2))

        self.dec_conv_20 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_2,
                               out_channels=channel_depth_1,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_1))

        # Decoder Hidden Layer 1
        self.dec_conv_11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_1,
                               out_channels=channel_depth_1,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_1))

        self.dec_conv_10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_1,
                               out_channels=channel_depth_0,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_0))

        # Decoder Hidden Layer 0
        self.dec_conv_01 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_depth_0,
                               out_channels=channel_depth_0,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(channel_depth_0))

        self.dec_conv_00 = nn.ConvTranspose2d(in_channels=channel_depth_0,
                                              out_channels=self.num_classes,
                                              kernel_size=3,
                                              padding=1)

        # Final upsampling to match target size
        # self.upsample = nn.modules.UpsamplingNearest2d(size=size)

        if self.pretrained:
            self.init_weigts()

    def init_weigts(self):
        vgg = models.vgg16(pretrained=True)

        # Hidden Layer 0
        self.enc_conv_00[0].weight.data - vgg.features[0].weight.data
        self.enc_conv_00[0].bias.data - vgg.features[0].bias.data

        self.enc_conv_01[0].weight.data - vgg.features[2].weight.data
        self.enc_conv_01[0].bias.data - vgg.features[2].bias.data

        # Hidden Layer 1
        self.enc_conv_10[0].weight.data - vgg.features[5].weight.data
        self.enc_conv_10[0].bias.data - vgg.features[5].bias.data

        self.enc_conv_11[0].weight.data - vgg.features[7].weight.data
        self.enc_conv_11[0].bias.data - vgg.features[7].bias.data

        # Hidden Layer 2
        self.enc_conv_20[0].weight.data - vgg.features[10].weight.data
        self.enc_conv_20[0].bias.data - vgg.features[10].bias.data

        self.enc_conv_21[0].weight.data - vgg.features[12].weight.data
        self.enc_conv_21[0].bias.data - vgg.features[12].bias.data

        self.enc_conv_22[0].weight.data - vgg.features[14].weight.data
        self.enc_conv_22[0].bias.data - vgg.features[14].bias.data

        # Hidden Layer 3
        self.enc_conv_30[0].weight.data - vgg.features[17].weight.data
        self.enc_conv_30[0].bias.data - vgg.features[17].bias.data

        self.enc_conv_31[0].weight.data - vgg.features[19].weight.data
        self.enc_conv_31[0].bias.data - vgg.features[19].bias.data

        self.enc_conv_32[0].weight.data - vgg.features[21].weight.data
        self.enc_conv_32[0].bias.data - vgg.features[21].bias.data

        # Hidden Layer 4
        self.enc_conv_40[0].weight.data - vgg.features[24].weight.data
        self.enc_conv_40[0].bias.data - vgg.features[24].bias.data

        self.enc_conv_41[0].weight.data - vgg.features[26].weight.data
        self.enc_conv_41[0].bias.data - vgg.features[26].bias.data

        self.enc_conv_42[0].weight.data - vgg.features[28].weight.data
        self.enc_conv_42[0].bias.data - vgg.features[28].bias.data

    def forward(self, image, size):
        """
        Forward pass
        """

        # Encoding Layer 0
        size_0 = image.size()
        x = F.relu(self.enc_conv_00(image))
        x = F.relu(self.enc_conv_01(x))
        x, ind_0 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoding Layer 1
        size_1 = x.size()
        x = F.relu(self.enc_conv_10(x))
        x = F.relu(self.enc_conv_11(x))
        x, ind_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoding Layer 2
        size_2 = x.size()
        x = F.relu(self.enc_conv_20(x))
        x = F.relu(self.enc_conv_21(x))
        x = F.relu(self.enc_conv_22(x))
        x, ind_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoding Layer 3
        size_3 = x.size()
        x = F.relu(self.enc_conv_30(x))
        x = F.relu(self.enc_conv_31(x))
        x = F.relu(self.enc_conv_32(x))
        x, ind_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoding Layer 4
        size_4 = x.size()
        x = F.relu(self.enc_conv_40(x))
        x = F.relu(self.enc_conv_41(x))
        x = F.relu(self.enc_conv_42(x))
        x, ind_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Decoder Layer 4
        x = F.max_unpool2d(x, ind_4, kernel_size=2, stride=2, output_size=size_4)
        x = F.relu(self.dec_conv_42(x))
        x = F.relu(self.dec_conv_41(x))
        x = F.relu(self.dec_conv_40(x))

        # Decoder Layer 3
        x = F.max_unpool2d(x, ind_3, kernel_size=2, stride=2, output_size=size_3)
        x = F.relu(self.dec_conv_32(x))
        x = F.relu(self.dec_conv_31(x))
        x = F.relu(self.dec_conv_30(x))

        # Decoder Layer 2
        x = F.max_unpool2d(x, ind_2, kernel_size=2, stride=2, output_size=size_2)
        x = F.relu(self.dec_conv_22(x))
        x = F.relu(self.dec_conv_21(x))
        x = F.relu(self.dec_conv_20(x))

        # Decoder Layer 1
        x = F.max_unpool2d(x, ind_1, kernel_size=2, stride=2, output_size=size_1)
        x = F.relu(self.dec_conv_11(x))
        x = F.relu(self.dec_conv_10(x))

        # Decoder Layer 0
        x = F.max_unpool2d(x, ind_0, kernel_size=2, stride=2, output_size=size_0)
        x = F.relu(self.dec_conv_01(x))
        x = self.dec_conv_00(x)

        # Upsample to match target
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return F.softmax(x, dim=1)

    def loss(self, image, target):
        loss = self.loss_fct(image, target)

        return loss
