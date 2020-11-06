# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

from torchvision import transforms
from utils.transform import conv_downsample
from predictors.basepredictor import BasePredictor
from utils.dataloader import custom_dataloaders


class SegNet_Predictor(BasePredictor):
    """
    Predictor for Fast SCNN model
        Inherited from BasePredictor
    """

    def __init__(self, model, config, experiment_dir):
        """
        Initialize all the dirs for predictions.
        :param model: model to use for predictions.
        :param config: dictionary containing the configuration.
        """
        super(SegNet_Predictor, self).__init__(model, config, experiment_dir)

        # Creating the transforms
        transform = transforms.Compose([
            conv_downsample(model.num_classes, factor=4, device=self.device),
        ])

        # Creating the dataloaders
        _, _, _, predict_loader = custom_dataloaders(config['data'], self.device,
                                                     transforms=transform)

        self.predict_loader = predict_loader
        self.labels = predict_loader.dataset.labels

        print("Prediction Dataset loaded.")
        print(" %3d batches" % (len(predict_loader)))
