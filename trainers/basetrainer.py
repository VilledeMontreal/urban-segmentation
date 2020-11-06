# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import json
import torch
import mlflow
import logging
from time import time
from utils.utils import set_seeds
from tensorboardX import SummaryWriter
from utils.metrics import evaluate_accuracy, evaluate_iou, clean_iou


logging.basicConfig(level=logging.INFO, format='')


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, config, resume,
                 experiment_dir, hyperparams=None):
        """
        Initialize all the directories and logging for a training.
        :param model: model to train.
        :param optimizer: optimizer to use for training.
        :param config: dictionary containing the configuration.
        :param resume: path to a checkpoint to resume training.
        :param experiment_dir: optional argument for where to log
        and save checkpoints (used for hyperparamter search).
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # GPU device if available
        if torch.cuda.is_available():
            print("Using the GPU")
            self.device = torch.device("cuda")
        else:
            print("WARNING: You are about to run on cpu.")
            self.device = torch.device("cpu")

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.log_period = cfg_trainer['log_period']
        self.save_period = cfg_trainer['save_period']

        if config['data']['dataset']['labels2keep']:
            self.classes = config['data']['dataset']['labels2keep']
        else:
            self.classes = None

        # Tensorboard logs
        self.log_dir = os.path.join(cfg_trainer['log_dir'], config['name'],
                                    experiment_dir)
        self.tb_writer = SummaryWriter(logdir=self.log_dir)

        set_logger(self.logger, os.path.join(self.log_dir, "file.log"))

        self.model = model.to(self.device)

        self.tb_writer.add_text('model', str(self.model))
        self.tb_writer.add_text('config', json.dumps(config))

        self.optimizer = optimizer

        self.start_epoch = 0

        self.best_loss = 9999
        self.best_accuracy = 0
        self.best_iou = 0

        self.saved_epochs = {}
        self.saved_epochs['best_iou'] = 0
        self.saved_epochs['best_acc'] = 0
        self.saved_epochs['best_loss'] = 0
        self.saved_epochs['last_checkpt'] = 0
        self.consecutive_stale = 0
        self.consecutive_stale_break = cfg_trainer['break_epoch']

        # Save configuration file into logs directory:
        config_save_path = os.path.join(self.log_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        # Setting up MLFlow
        mlflow.set_tracking_uri('file:' + cfg_trainer['log_dir'] + 'mlruns')
        mlflow.set_experiment(config['name'])

        if resume:
            self._resume_checkpoint(resume)
            self.mlflow_run = mlflow.start_run(run_id=self.mlflow_runid)
        else:
            self.mlflow_run = mlflow.start_run(nested=True)

            mlflow.log_param('Dataset', config['data']['dataset']['name'])
            mlflow.log_param('lr', self.optimizer.defaults['lr'])
            if self.classes:
                mlflow.log_param('Classes', self.classes)
            else:
                mlflow.log_param('Classes', 'All')

        with open(os.path.join(self.log_dir, 'saved_epochs.txt'), 'w') as jsf:
            json.dump(self.saved_epochs, jsf)

    def train(self):
        """
        Full training logic
        """
        t0 = time()

        for epoch in range(self.start_epoch, self.epochs):
            set_seeds(self.config['seeds'], epoch)

            # Train and Valid losses
            train_loss = self._train_epoch(epoch)
            valid_loss, accuracy, iou, _ = self._valid_epoch(epoch)

            time_elapsed = time() - t0
            print("Epoch %1d completed after %4d secs" % (epoch, time_elapsed))

            self._save_checkpoint(epoch, train_loss, valid_loss, accuracy, iou,
                                  save_last=True)

            if self.consecutive_stale >= self.consecutive_stale_break:
                break

        mlflow.log_metric('Loss', self.best_loss)
        mlflow.log_metric('IoU', self.best_iou)
        mlflow.log_metric('Accuracy', self.best_accuracy)
        mlflow.log_metric('Epoch', epoch)

        mlflow.end_run()

        return self.best_iou

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch.
        :return: the loss for this epoch
        """

        total_loss = 0
        self.model.train()

        self.logger.info('Train Epoch: {}'.format(epoch))

        for batch_id, data in enumerate(self.train_loader):
            start_time = time()

            image = data['image']
            target = data['target']

            self.model.zero_grad()

            prediction = self.model(image, target.shape[2:4])

            loss = self.model.loss(prediction, target)

            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.train_loader) + batch_id

            self.tb_writer.add_scalar('train/loss', loss.item(), step)

            total_loss += loss.item()

            end_time = time()
            it_time = end_time - start_time

            if batch_id % self.log_period == 0:
                accuracy = evaluate_accuracy(prediction, target)
                self.tb_writer.add_scalar('train/accuracy', accuracy, step)
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} Accuracy: {:.2f}'.format(
                        batch_id * self.train_loader.batch_size + image.size(0),
                        len(self.train_loader.dataset),
                        100.0 * batch_id / len(self.train_loader),
                        it_time * (len(self.train_loader) - batch_id),
                        loss.item(),
                        accuracy))

        self.logger.info('   > Total loss: {:.6f}'.format(
            total_loss / len(self.train_loader)
        ))

        return total_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch
        :param epoch: Current epoch.
        :return: the loss for this epoch
        """

        total_loss = 0
        total_accuracy = 0
        iou = 0
        exist = 0

        self.model.eval()

        self.logger.info('Valid Epoch: {}'.format(epoch))

        for batch_id, data in enumerate(self.valid_loader):
            start_time = time()

            image = data['image']
            target = data['target']

            prediction = self.model(image, target.shape[2:4])

            loss = self.model.loss(prediction, target)

            accuracy = evaluate_accuracy(prediction, target)

            _iou, _exist = evaluate_iou(prediction, target, self.model.num_classes)
            iou += _iou
            exist += _exist

            step = epoch * len(self.valid_loader) + batch_id

            self.tb_writer.add_scalar('valid/loss', loss.item(), step)
            self.tb_writer.add_scalar('valid/accuracy', accuracy, step)

            total_loss += loss.item()
            total_accuracy += accuracy

            end_time = time()
            it_time = end_time - start_time

            if batch_id % self.log_period == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} Accuracy: {:.2f}'.format(
                        batch_id * self.valid_loader.batch_size + image.size(0),
                        len(self.valid_loader.dataset),
                        100.0 * batch_id / len(self.valid_loader),
                        it_time * (len(self.valid_loader) - batch_id),
                        loss.item(),
                        accuracy))

        L_val = total_loss / len(self.valid_loader)
        acc = total_accuracy / len(self.valid_loader)
        mean_iou, iou_data = clean_iou(iou, exist)

        self.tb_writer.add_scalar('valid/iou', accuracy, epoch)

        self.logger.info(
            '   > [Metrics for epoch {}] Loss: {:.6f} - Accuracy: {:.2f} - IoU: {:.2f}'.format(
                epoch,
                L_val,
                acc,
                mean_iou))

        return L_val, acc, mean_iou, iou_data

    def make_prediction(self, image):
        """
        Generate class prediction for a given image
        :param image: The image used for predictions
        Returns the predicted class for each pixel of the image
        """
        prediction = torch.argmax(image, dim=1)

        return prediction

    def _save_checkpoint(self, epoch, train_loss, valid_loss, accuracy, iou,
                         save_last=False):
        """
        Saving checkpoints. The regular checkpoint is save only if
        the epoch is within the save period.
        If the current model is the best performing, it is saved as the
        best_checkpoint, regardless of the save period.
        :param epoch: current epoch number
        :param train_loss: current training loss
        :param valid_loss: current validation loss
        """
        save = save_last
        improved = False

        arch = type(self.model).__name__

        if self.best_loss > valid_loss or epoch == 0:
            self.best_loss = valid_loss
            save = True
            improved = True

        if self.best_accuracy < accuracy or epoch == 0:
            self.best_accuracy = accuracy
            save = True
            improved = True

        if self.best_iou < iou or epoch == 0:
            self.best_iou = iou
            save = True
            improved = True

        if improved is True:
            self.consecutive_stale = 0
        else:
            self.consecutive_stale += 1

        if save is True:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'best_iou': self.best_iou,
                'best_accuracy': self.best_accuracy,
                'best_loss': self.best_loss,
                'consecutive_stale': self.consecutive_stale,
                'saved_epochs': self.saved_epochs,
                'mlflow_runid': self.mlflow_run.info.run_id
            }

        if self.best_loss == valid_loss:
            filename_best = os.path.join(self.log_dir, 'best_loss.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best loss checkpoint")
            self.saved_epochs['best_loss'] = epoch

        if self.best_accuracy == accuracy:
            filename_best = os.path.join(self.log_dir, 'best_acc.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best accuracy checkpoint.")
            self.saved_epochs['best_acc'] = epoch

        if self.best_iou == iou:
            filename_best = os.path.join(self.log_dir, 'best_iou.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best iou checkpoint")
            self.saved_epochs['best_iou'] = epoch

        if save_last is True:
            filename = os.path.join(self.log_dir, 'last_checkpoint.pth')
            torch.save(state, filename)
            self.logger.info("Saving last checkpoint")
            self.saved_epochs['last_checkpt'] = epoch

        if save is True:
            with open(os.path.join(self.log_dir, 'saved_epochs.txt'), 'w') as jsf:
                json.dump(self.saved_epochs, jsf)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is '
                'different from that of checkpoint. '
                'This may yield an exception while state_dict '
                'is being loaded.')

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint
        # only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != \
                self.config['optimizer']['type']:
            self.logger.warning(
                'Warning: Optimizer type given in config file is '
                'different from that of checkpoint. '
                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint '{}' (epoch {}) loaded".format(resume_path,
                                                       self.start_epoch))

        # Load the best metrics
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_loss = checkpoint['best_loss']
        self.best_iou = checkpoint['best_iou']
        self.consecutive_stale = checkpoint['consecutive_stale']

        self.saved_epochs = checkpoint['saved_epochs']

        self.mlflow_runid = checkpoint['mlflow_runid']


def set_logger(logger, log_file):
    """
    Set a file handler to write the logs to a file.
    :param logger: the logger to use.
    :param log_file: the file where to write the logs.
    """
    # Create handlers
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)

    # Create formatter and add it to handler
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
