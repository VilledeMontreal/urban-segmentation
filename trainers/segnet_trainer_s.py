# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

from trainers.basetrainer_s import BaseTrainer_S
from utils.dataloader import custom_dataloaders
from torchvision import transforms
from utils.transform import conv_downsample


class SegNet_Trainer_S(BaseTrainer_S):
    """
    Trainer for Fast SCNN model
        Inherited from BaseTrainer
    """

    def __init__(self, model, model_d, optimizer, optimizer_d, config, resume,
                 experiment_dir, hyperparams=None, shrink_factor=1):
        """
        Initialize the trainer.
        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param experiment_dir: optional argument for where to log
            and save checkpoints (used for hyperparamter search).
        """
        super(SegNet_Trainer_S, self).__init__(model, model_d, optimizer, optimizer_d,
                                               config, resume, experiment_dir,
                                               hyperparams)

        # Creating the dataloaders
        if shrink_factor == 1:
            train_l, valid_l, _, extra_l, _ = custom_dataloaders(config['data'],
                                                                 self.device, extra=True)

        else:
            transform = transforms.Compose([
                conv_downsample(model.num_classes, factor=shrink_factor,
                                device=self.device),
            ])

            train_l, valid_l, _, extra_l, _ = custom_dataloaders(config['data'],
                                                                 self.device,
                                                                 extra=True,
                                                                 transforms=transform)

        self.train_loader = train_l
        self.valid_loader = valid_l
        self.extra_loader = extra_l

        self.train_loader_iterator = iter(train_l)
        self.valid_loader_iterator = iter(valid_l)
        self.extra_loader_iterator = iter(extra_l)

        print(config['data']['dataset']['name'] + " Dataset loaded.")
        print(" Train | Extra ")
        print(" %5d ¦ %6d batches" % (len(train_l), len(extra_l)))
