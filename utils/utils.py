# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import numpy as np
import random
import torch


def set_seeds(config, offset=0):
    np.random.seed(config['numpy'] + offset)
    torch.manual_seed(config['torch'] + offset)
    random.seed(config['random'] + offset)


def one_hot(x, num_classes, device):
    """
    One-hot encoding of a 2D image
    :param x: A batch of 2D images (B x H x W)
    :param num_class: The number of classes to encode
    Returns One-hot encoded tensor B x C x H x W
    """

    one_hot = np.zeros((num_classes, x.shape[0], x.shape[1]))

    for c in range(num_classes):
        one_hot[c, :, :] = (x == c)

    one_hot = torch.FloatTensor(one_hot).to(device)

    return one_hot
