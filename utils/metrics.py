# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import torch
import numpy as np


def evaluate_iou(prediction, target, num_classes, classes=None):
    """
    Intersection of prediction and target over Union of prediction and target
    If classes is given, the classes that are not included are changed to void
    """

    _, pred = torch.max(prediction, 1)
    _, target = torch.max(target, 1)

    iou = []
    exist = []

    for c in range(num_classes):
        A = pred == c
        B = target == c

        intersection = torch.min(A, B)  # Logical And
        union = torch.max(A, B)         # Logical Or

        total_union = torch.sum(union).float()

        if total_union > 0:
            iou.append(torch.sum(intersection).float() / total_union)
            exist.append(1)
        else:
            iou.append(0.)
            exist.append(0)

    return np.array(iou), np.array(exist)


def clean_iou(iou, exist, classes=None):
    clean_iou = np.array([])
    iou_data = {}

    if classes is None:
        classes = range(len(iou))

    for i, c in enumerate(classes):
        if exist[i] > 0:
            clean_iou = np.append(clean_iou, iou[i].item() / exist[i].item())
            iou_data[c] = clean_iou[-1]

        else:
            iou_data[c] = "N/A"

    return np.mean(clean_iou), iou_data


def evaluate_accuracy(prediction, target):
    """
    Pixelwise Accuracy of Prediction
    """

    _, pred = torch.max(prediction, 1)
    _, target = torch.max(target, 1)

    evaluation = pred == target
    correct = torch.sum(evaluation).type(torch.DoubleTensor).item()
    total = target.shape[0] * target.shape[1] * target.shape[2]

    return correct / total


if __name__ == "__main__":

    classes = [0, 1, 2, 5]
    prediction = torch.Tensor((
        [[[[0, 0, 0], [0, 0, 0]],
          [[0, 1, 0], [0, 0, 0]],
          [[1, 0, 1], [0, 0, 0]],
          [[0, 0, 0], [1, 0, 0]],
          [[0, 0, 0], [0, 1, 0]],
          [[0, 0, 0], [0, 0, 1]]]]
    ))

    target = torch.Tensor((
        [[[1, 1, 2], [2, 4, 5]]]
    ))

    iou, exist = evaluate_iou(prediction, target, 6, 'cpu', classes)

    mean, data = clean_iou(iou, exist, classes)

    print('hello')
