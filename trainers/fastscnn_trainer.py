# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.


from trainers.basetrainer import BaseTrainer
from utils.dataloader import custom_dataloaders


class FastSCNN_Trainer(BaseTrainer):
    """
    Trainer for Fast SCNN model
        Inherited from BaseTrainer
    """

    def __init__(self, model, optimizer, config, resume, experiment_dir,
                 hyperparams=None):
        """
        Initialize the trainer.
        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param resume: path to a checkpoint to resume training.
        :param config: dictionary containing the configuration.
        :param experiment_dir: optional argument for where to log
            and save checkpoints (used for hyperparamter search).
        """
        super(FastSCNN_Trainer, self).__init__(model, optimizer, config, resume,
                                               experiment_dir, hyperparams)

        # Creating the dataloaders
        train_loader, valid_loader, _, _ = custom_dataloaders(config['data'], self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        print(config['data']['dataset']['name'] + " Dataset loaded.")
        print(" Train | Valid ")
        print(" %5d ¦ %6d batches" % (len(train_loader), len(valid_loader)))
