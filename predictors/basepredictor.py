# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import torch
import numpy as np
from time import time
from PIL import Image
from utils.utils import set_seeds
from utils.test_package import show_tensor


class BasePredictor:
    """
    Base class for all predictors
    """

    def __init__(self, model, config, experiment_dir):
        """
        Initialize all the dirs for predictions.
        :param model: model to use for predictions.
        :param config: dictionary containing the configuration.
        """
        self.config = config
        cfg_pred = config['predictor']
        self.red_shade = cfg_pred['red_shade']
        self.rgb_label = cfg_pred['rgb_label']

        self.save_dir = os.path.join(cfg_pred['save_dir'], config['name'], experiment_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, "Redshade"))
            os.makedirs(os.path.join(self.save_dir, "RGB"))
        else:
            if not os.path.exists(os.path.join(self.save_dir, "Redshade")):
                os.makedirs(os.path.join(self.save_dir, "Redshade"))

            if not os.path.exists(os.path.join(self.save_dir, "RGB")):
                os.makedirs(os.path.join(self.save_dir, "RGB"))

            if not os.path.exists(os.path.join(self.save_dir, "RGB", "Input")):
                os.makedirs(os.path.join(self.save_dir, "RGB", "Input"))

        # GPU device if available
        if torch.cuda.is_available():
            print("Using the GPU")
            self.device = torch.device("cuda")
        else:
            print("WARNING: You are about to run on cpu.")
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        model.eval()
        torch.no_grad()

    def predict(self):
        t0 = time()

        set_seeds(self.config['seeds'])

        for batch_id, sample in enumerate(self.predict_loader):
            image = sample['image']
            image = image.to(self.device)

            name = sample['name']
            initial_size = sample['initial_size']

            prediction = self.make_prediction(image)

            self.save_prediction(prediction, name, initial_size, image)

        time_elapsed = time() - t0
        print("Prediction completed after %4d secs" % (time_elapsed))

    def make_prediction(self, image):
        output = self.model(image, image.shape[2:4])
        prediction = torch.argmax(output, dim=1)

        return prediction

    def save_prediction(self, prediction, name, initial_size, _input=None):
        for i in range(prediction.size(0)):
            img = prediction[i, :, :]

            if self.device != "cpu":
                img = img.cpu()

            imgDataArr = img.data.numpy()

            if self.red_shade:
                imgRArr = np.zeros((initial_size[0][i], initial_size[1][i], 3),
                                   dtype="uint8")

                imgRArr[:, :, 0] = imgDataArr[0:initial_size[0][i], 0:initial_size[1][i]]

                imgRGB = Image.fromarray(imgRArr)
                imgRGB.save(os.path.join(self.save_dir, "Redshade", name[i]))

            if self.rgb_label:
                imgRGBArray = np.zeros((initial_size[0][i], initial_size[1][i], 3),
                                       dtype="uint8")

                for label in range(len(self.labels)):
                    ix = imgDataArr[0:initial_size[0][i], 0:initial_size[1][i]] == label
                    imgRGBArray[ix] = self.labels[label][2:5]

                imgRGB = Image.fromarray(imgRGBArray)
                imgRGB.save(os.path.join(self.save_dir, "RGB", name[i]))

            if _input is not None:
                in_img = _input[i]
                show_tensor(in_img, os.path.join(self.save_dir, "RGB", "Input"), name[i])
