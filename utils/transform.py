# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        swap color axis because
        numpy image: H x W x C
        torch image: C X H X W
    """

    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return sample


class dummy_downsample(object):
    """
    Downsample an image by taking every factor-th pixel.
    """

    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, sample):
        if len(sample) == 2:
            sample = list(sample)
            sample[0] = sample[0][:, 0::self.factor, 0::self.factor]
            sample[1] = sample[1][:, 0::self.factor, 0::self.factor]

            return tuple(sample)

        else:
            sample = sample[:, 0::self.factor, 0::self.factor]

            return sample


class functional_fixed_size(object):
    """
    Upsamples the image and target to the specified size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if len(sample) == 2:
            sample = list(sample)
            sample[0] = F.interpolate(sample[0].unsqueeze_(0), size=self.size,
                                      mode='bilinear', align_corners=True)
            sample[0].squeeze_(0)
            sample[1] = F.interpolate(sample[1].unsqueeze_(0), size=self.size,
                                      mode='nearest')
            sample[1].squeeze_(0)

            return tuple(sample)

        else:
            sample = F.interpolate(sample.unsqueeze_(0), size=self.size,
                                   mode='bilinear', align_corners=True)
            sample.squeeze_(0)

            return sample


class conv_downsample(object):
    """
    Downsample an image and the target by averaging with a fixed kernel
    """

    def __init__(self, num_classes, factor=2, device='cpu'):
        self.factor = factor
        self.device = device
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=factor,
            stride=factor,
            groups=3,
            bias=False)

        kernel = torch.ones((3, 1, factor, factor)) / (factor**2)

        self.conv.weight = nn.Parameter(kernel)
        self.conv.weight.requires_grad = False
        self.conv.to(device)

    def __call__(self, sample):
        if len(sample) == 2:
            sample = list(sample)
            sample[0] = self.conv(sample[0].unsqueeze_(0)).squeeze_(0)

            return tuple(sample)

        else:
            sample = self.conv(sample.unsqueeze_(0)).squeeze_(0)

            return sample


class dummy_upsample(object):
    """
    Simple upsample by scaling every pixel to factor x factor
    """

    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, image):
        size = (image.shape[0],) + (self.factor * image.shape[1],)
        size = size + (self.factor * image.shape[2],)
        newImage = torch.empty(size, dtype=torch.uint8)

        for f1 in range(self.factor):
            for f2 in range(self.factor):
                newImage[:, f1::self.factor, f2::self.factor] = image

        return newImage


class DefaultTransform(object):
    """
    Default Carla Transforms
    """

    def __init__(self, device):
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

    def __call__(self, sample):
        return self.transform(sample).to(self.device)


class InitialCropPad(object):
    """
    Initial Transform to crop or pad the image to a constant size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, initial_size, target=None):
        if target is None:
            newImage = self.cropPad(image, initial_size)

            return newImage.to(image.device)

        else:
            newImage = self.cropPad(image, initial_size)
            newTarget = self.cropPad(target, initial_size)

            return newImage.to(image.device), newTarget.to(target.device)

    def cropPad(self, image, initial_size):
        tempImage = torch.zeros((image.shape[0], initial_size[0], self.size[1]))
        newImage = torch.zeros((image.shape[0], self.size[0], self.size[1]))

        # Width
        if self.size[1] > initial_size[1]:
            tempImage[:, :, int((self.size[1] - initial_size[1]) / 2):int((self.size[1] + initial_size[1]) / 2)] = image
        else:
            tempImage = image[:, :, int((initial_size[1] - self.size[1]) / 2):int((self.size[1] + initial_size[1]) / 2)]

        # Height
        if self.size[0] > initial_size[0]:
            newImage[:, int((self.size[0] - initial_size[0]) / 2):int((self.size[0] + initial_size[0]) / 2), :] = tempImage
        else:
            newImage = tempImage[:, int((initial_size[0] - self.size[0]) / 2):int((self.size[0] + initial_size[0]) / 2), :]

        return newImage


class MinimalTargets(object):
    """
    Changes unwanted labels in target to void.
    """

    def __init__(self, list_to_keep=[0]):
        self.list_to_keep = list_to_keep

    def __call__(self, target):
        for label in np.unique(target):
            if label not in self.list_to_keep:
                target[target == label] = 0

        i = 0
        for label in self.list_to_keep:
            target[target == label] = i
            i += 1

        return target
