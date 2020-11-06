# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

from trainers.basetrainer import BaseTrainer
from utils.dataloader import custom_dataloaders
from torchvision import transforms
from utils.transform import conv_downsample


class SegNet_Trainer(BaseTrainer):
    """
    Trainer for Fast SCNN model
        Inherited from BaseTrainer
    """

    def __init__(self, model, optimizer, config, resume, experiment_dir,
                 hyperparams=None, shrink_factor=1):
        """
        Initialize the trainer.
        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param experiment_dir: optional argument for where to log
            and save checkpoints (used for hyperparamter search).
        """
        super(SegNet_Trainer, self).__init__(model, optimizer, config, resume,
                                             experiment_dir, hyperparams)

        # Creating the dataloaders
        if shrink_factor == 1:
            train_l, valid_l, _, _ = custom_dataloaders(config['data'], self.device)

        else:
            transform = transforms.Compose([
                conv_downsample(model.num_classes, factor=shrink_factor,
                                device=self.device),
            ])

            train_l, valid_l, _, _ = custom_dataloaders(config['data'], self.device,
                                                        transforms=transform)

        self.train_loader = train_l
        self.valid_loader = valid_l

        print(config['data']['dataset']['name'] + " Dataset loaded.")
        print(" Train | Valid ")
        print(" %5d ¦ %5d batches" % (len(train_l), len(valid_l)))
