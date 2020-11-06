# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from labels import CGMU_LABELS

W = 704
H = 480


def make_heatmap(label, folder, threshold=None):
    priv = np.zeros((H, W))

    for _, _, images in os.walk(folder):
        nb_imgs = len(images)
        for im in images:
            img = np.array(Image.open(os.path.join(folder, im)))[:, :, 0:3]

            r = img[:, :, 0] == CGMU_LABELS[label][2]
            g = img[:, :, 1] == CGMU_LABELS[label][3]
            b = img[:, :, 2] == CGMU_LABELS[label][4]

            first_and = np.logical_and(r, g)

            priv += np.logical_and(first_and, b)

    if np.max(priv) == 0:
        print('No label ' + CGMU_LABELS[label][0])

    else:
        if threshold:
            priv = priv / np.max(priv)
            priv[np.where(priv > threshold)] = 0

        else:
            priv[0, 0] = nb_imgs

        plt.imshow(priv, cmap='hot')
        plt.axis('off')
        plt.title(CGMU_LABELS[label][0])
        plt.show()


SET = '4_'

if __name__ == "__main__":
    for i in range(13):
        make_heatmap(i, 'heatmap', threshold=0.85)
