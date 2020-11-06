# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import torch
import numpy as np
from PIL import Image


def show_tensor(x, save_path, name, labels=None):
    """
    Save Image Tensors to disk
    Inputs:
        x:          Image Tensor [Channel, Height, Width]
                    or Image Batch [Batch, Channel, Height, Width]
        save_path:  Path to directory where images are saved
        name:       Default name of images
        labels:     If specified, treat images of x as one-hot encoded
                    Must be list of tuples for each class
                    ie.: [('void', 0, R, G, B]), ('class1', 1, R, G, B]), ...]
    """

    # Create save directory if needed
    if os.path.isdir(save_path) is True:
        offset = len(os.listdir(save_path))

    else:
        os.mkdir(save_path)
        offset = 0

    # Transform Image Tensors to PIL Images
    if len(x.shape) == 3:
        images = [_conversion_function(labels)(x, labels)]

    elif len(x.shape) == 4:
        images = []
        for batch in range(x.shape[0]):
            images.append(_conversion_function(labels)(x[batch], labels))

    else:
        print("Input size mismatch")

    # Save PIL Images
    for i in range(len(images)):
        fn = ("Sem_" if labels else "RGB_") + name + str(i + offset).zfill(5) + ".png"
        image_filepath = os.path.join(save_path, fn)
        images[i].save(image_filepath)


def _RGB_tensor_to_PIL(x, labels=None):
    # Change dimensions to be [Height, Width, Channel]
    x = 255 * x.transpose_(0, 1).transpose_(2, 1)

    # Transform tensor into numpy array
    x = x.cpu().numpy().astype('uint8')

    # Tranform numpy array into PIL Image
    x = Image.fromarray(x)

    return x


def _Semantic_tensor_to_PIL(y, labels):
    # Get classes from one_hot encoding
    _, y = torch.max(y, dim=0)

    # Transform tensor into numpy array
    y = y.cpu().numpy().astype('uint8')

    # Convert classes to RGB
    y_RGB = np.zeros(y.shape + (3,), dtype='uint8')

    for label in labels:
        y_RGB[y == label[1]] = [label[2:5]]

    # Transform numpy array into PIL Image
    y = Image.fromarray(y_RGB)

    return y


def _conversion_function(labels):
    if labels is None:
        return _RGB_tensor_to_PIL

    else:
        return _Semantic_tensor_to_PIL


def test_for_corruption(loader):
    """
    Run through a dataloader once to make sure all images load
    """
    loader_iterator = iter(loader)
    i = 0

    while(True and i < 22000):
        i += 1
        try:
            _ = loader_iterator.next()

        except StopIteration:
            print("Went through all images succesfully")
            break


if __name__ == "__main__":
    from utils.load_image import load_pair

    path1 = "../Predictions/CGMU_L/Fast_SCNN_CGMU_L/1213_114628/RGB/001226.jpeg"
    path2 = "../Predictions/CGMU_L/Fast_SCNN_CGMU_L/1213_114628/RGB/Input/RGB_001226.jpeg00000.png"
    load_pair(path1, path2)
