# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import json
import torch
import mlflow
import logging
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import set_seeds
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.metrics import evaluate_accuracy, evaluate_iou, clean_iou


logging.basicConfig(level=logging.INFO, format='')


class BaseTrainer_S:
    """
    Base class for all adversarial trainers
    """

    def __init__(self, model, model_d, optimizer, optimizer_d, config, resume,
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
        self.nb_iters = cfg_trainer['nb_iters']
        self.semi_start = cfg_trainer['semi_start']
        self.save_period = cfg_trainer['save_period']
        self.nb_valid_iters = cfg_trainer['nb_valid_iters']

        if hyperparams:
            self.threshold = torch.Tensor([hyperparams['threshold']]).to(self.device)
            self.lambda_adv_label = hyperparams['lambda_adv_label']
            self.lambda_adv_unlabel = hyperparams['lambda_adv_unlabel']
            self.lambda_semi = hyperparams['lambda_semi']
        else:
            self.threshold = torch.Tensor([cfg_trainer['threshold']]).to(self.device)
            self.lambda_adv_label = cfg_trainer['lambda_adv_label']
            self.lambda_adv_unlabel = cfg_trainer['lambda_adv_unlabel']
            self.lambda_semi = cfg_trainer['lambda_semi']

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
        self.model_d = model_d.to(self.device)

        self.tb_writer.add_text('model', str(self.model))
        self.tb_writer.add_text('config', json.dumps(config))

        self.optimizer = optimizer
        self.optimizer_d = optimizer_d

        self.start_iter = 0

        self.best_loss = 9999
        self.best_iou = 0
        self.best_accuracy = 0

        self.saved_iter = {}
        self.saved_iter['best_iou'] = 0
        self.saved_iter['best_acc'] = 0
        self.saved_iter['best_loss'] = 0
        self.saved_iter['last_checkpt'] = 0

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

            mlflow.log_param('Nb iters', self.nb_iters)
            mlflow.log_param('Semi-start iter', self.semi_start)
            mlflow.log_param('Threshold', self.threshold)
            mlflow.log_param('Lambda adv labelled', self.lambda_adv_label)
            mlflow.log_param('Lambda adv unlabelled', self.lambda_adv_unlabel)
            mlflow.log_param('Lambda semi', self.lambda_semi)
            mlflow.log_param('Dataset', config['data']['dataset']['name'])
            mlflow.log_param('Lr', config['optimizer']['args']['lr'])
            if self.classes:
                mlflow.log_param('Classes', self.classes)
            else:
                mlflow.log_param('Classes', 'All')

        with open(os.path.join(self.log_dir, 'saved_iter.txt'), 'w') as jsf:
            json.dump(self.saved_iter, jsf)

    def train(self):
        """
        Full training logic
        """

        t0 = time()

        for iteration in range(self.start_iter, self.nb_iters):
            set_seeds(self.config['seeds'], iteration)

            # Train and Valid losses
            L_semi, L_seg, L_disc = self._train_iter(iteration)

            iter_time = time() - t0

            if iteration % self.save_period == 0:
                train_loss = (L_semi, L_seg, L_disc)
                valid_loss, accuracy, iou, _ = self._valid_iter(iteration)

                self.logger.info(
                    '> [{}/{} ({:.0f}%), {:.2f}s] Semi_L: {:.6f} - Seg_L: {:.6f} - D_L: {:.6f}'.format(
                        iteration,
                        self.nb_iters,
                        100.0 * iteration / self.nb_iters,
                        iter_time * (self.nb_iters - iteration) /
                        (iteration - self.start_iter + 1),
                        L_semi,
                        L_seg,
                        L_disc))

                self.logger.info(
                    '> [Valid Loss: {:.6f} - Accuracy: {:.2f} - IoU: {:.3f} '.format(
                        valid_loss,
                        accuracy,
                        iou))

                # Save the checkpoints.
                time_elapsed = time() - t0
                print("Iteration %1d completed after %4d secs" %
                      (iteration, time_elapsed))
                self._save_checkpoint(iteration, train_loss, valid_loss,
                                      accuracy, iou, save_last=True)

        mlflow.log_metric('Loss', self.best_loss)
        mlflow.log_metric('IoU', self.best_iou)
        mlflow.log_metric('Accuracy', self.best_accuracy)

        mlflow.end_run()

        return self.best_iou

    def _train_iter(self, iteration):
        """
        Training logic for an iteration
        :param iteration: Current iteration.
        :return: the loss for this iteration
        """

        L_disc = 999  # Loss ofthe discriminator network
        L_seg = 999   # Loss of the segmentation network when training with labeled data
        L_semi = 999  # Unlabeled training loss

        self.model.train()
        self.model.zero_grad()

        self.model_d.eval()

        self._adjust_lr(self.optimizer_d, iteration, self.config['optimizer_d'])

        # Supervised Iteration
        try:
            sample_labeled = self.train_loader_iterator.next()
        except StopIteration:
            self.train_loader_iterator = iter(self.train_loader)
            sample_labeled = self.train_loader_iterator.next()

        image_labeled = sample_labeled['image']
        target = sample_labeled['target']

        L_seg, prediction_label = self._labeled_iteration(image_labeled, target)

        # Discriminator Iteration
        L_disc = self._discriminator_iteration(prediction_label, target)

        # Unsupervised Iteration
        if iteration >= self.semi_start:
            try:
                sample_unlabeled = self.extra_loader_iterator.next()
            except StopIteration:
                self.extra_loader_iterator = iter(self.extra_loader)
                sample_unlabeled = self.extra_loader_iterator.next()

            image_unlabeled = sample_unlabeled['image']

            L_semi = self._unlabeled_iteration(iteration, image_unlabeled,
                                               target.shape[2:4])

        self.tb_writer.add_scalar('train/L_semi', L_semi, iteration)
        self.tb_writer.add_scalar('train/L_seg', L_seg, iteration)
        self.tb_writer.add_scalar('train/L_disc', L_disc, iteration)

        return L_semi, L_seg, L_disc

    def _valid_iter(self, iteration):
        """
        Validation logic for an iteration
        :param iteration: Current iteration.
        :return: the loss for this iteration
        """

        L_val = 0
        accuracy = 0
        iou = 0
        exist = 0

        self.model.eval()

        for val_iter in range(self.nb_valid_iters):
            try:
                sample = self.valid_loader_iterator.next()

            except StopIteration:
                self.valid_loader_iterator = iter(self.valid_loader)
                sample = self.valid_loader_iterator.next()

            image_labeled = sample['image']
            target = sample['target']

            prediction = self.model(image_labeled, target.shape[2:4])

            L_seg = self._masked_half_log_loss(prediction, mask=target)
            L_val += L_seg.item()

            accuracy += evaluate_accuracy(prediction, sample['target'])

            _iou, _exist = evaluate_iou(prediction, sample['target'],
                                        self.model.num_classes, self.classes)
            iou += _iou
            exist += _exist

        L_val = L_val / self.nb_valid_iters
        accuracy = accuracy / self.nb_valid_iters
        mean_iou, iou_data = clean_iou(iou, exist, self.classes)

        self.tb_writer.add_scalar('valid/L_seg', L_val, iteration)
        self.tb_writer.add_scalar('valid/accuracy', accuracy, iteration)
        self.tb_writer.add_scalar('valid/mean_iou', mean_iou, iteration)

        return L_val, accuracy, mean_iou, iou_data

    def _unlabeled_iteration(self, iteration, image, size):
        prediction = self.model(image, size)

        confidence = self.model_d(prediction, size)

        L_adv = self._masked_half_log_loss(confidence)

        # Generate unsupervised mask
        mask = torch.zeros(confidence.shape).to(self.device)

        zeros = torch.zeros(confidence.shape).to(self.device)
        ones = torch.ones(confidence.shape).to(self.device)

        confidence_mask = torch.where(confidence > self.threshold, ones, zeros)

        _, prediction_argmax = torch.max(prediction, 1)
        prediction_argmax.unsqueeze_(1)

        for c in range(self.model.num_classes):
            argmax_mask = torch.where(prediction_argmax == c, ones, zeros)
            argmax_mask = torch.mul(argmax_mask, confidence_mask)
            mask = torch.cat((mask, argmax_mask), dim=1)

        mask = mask[:, 1:, :, :].detach()

        L_semi = self._masked_half_log_loss(prediction, mask=mask)

        loss = self.lambda_adv_unlabel * L_adv + self.lambda_semi * L_semi

        loss.backward()

        return loss.item()

    def _labeled_iteration(self, image, target):
        prediction = self.model(image, target.shape[2:4])
        prediction_label = prediction.detach()     # Used later to train the discriminator

        L_seg = self._masked_half_log_loss(prediction, mask=target)
        # L_seg = self._ce_loss(prediction, target)

        confidence = self.model_d(prediction, target.shape[2:4])

        L_adv = self._masked_half_log_loss(confidence)

        loss = L_seg + self.lambda_adv_label * L_adv

        loss.backward()

        self.optimizer.step()

        return loss.item(), prediction_label

    def _discriminator_iteration(self, prediction_label, target):
        self.model_d.train()
        self.model_d.zero_grad()

        self._toggle_disc_grad(True)

        # Train D from prediction
        confidence_pred = self.model_d(prediction_label, target.shape[2:4])

        L_disc_pred = self._masked_half_log_loss(confidence_pred, opposite=True) / 2
        L_disc_pred.backward()

        # Train D from label
        confidence_label = self.model_d(target, target.shape[2:4])

        L_disc_label = self._masked_half_log_loss(confidence_label) / 2
        L_disc_label.backward()

        # Add both losses
        loss = L_disc_pred + L_disc_label

        self.optimizer_d.step()

        self._toggle_disc_grad(False)

        return loss.item()

    def _toggle_disc_grad(self, toggle):
        for param in self.model_d.parameters():
            param.requires_grad = toggle

    def _semi_loss(self, confidence, label):
        target = np.ones(confidence.shape) * label
        target = Variable(torch.FloatTensor(target))
        target = target.to(self.device)

        loss = F.binary_cross_entropy(confidence, target)

        return loss

    def _ce_loss(self, prediction, target):
        loss_fct = nn.CrossEntropyLoss()

        batch_size = target.shape[0]

        mask = (target != 255)

        if mask.sum() == 0:
            return 0
        else:
            target = target[mask]
            target = target.view((batch_size, -1))

            large_mask = torch.stack([mask for i in range(self.model.num_classes)], 1)
            prediction = prediction[large_mask]
            prediction = prediction.view(batch_size, self.model.num_classes, -1)

            return loss_fct(prediction, target)

    def _half_log_loss(self, x, opposite=False):
        """
        x        : Tensor of size [B, H, W]
        opposite : If true, uses log(1-x). If false, uses log(x)
        """
        loss_fct = nn.BCELoss()

        if opposite:
            target = np.zeros(x.shape)
        else:
            target = np.ones(x.shape)

        target = Variable(torch.FloatTensor(target))
        target = target.to(self.device)

        return loss_fct(x, target)

    def _masked_half_log_loss(self, x, mask=None, opposite=False):
        """
        A variation of BCE to generate a masked log loss.
        The opposite argument filters either log(x) or log(1-x)
        The mask argument modifies elements of x to have their loss be 0
        x        : Tensor of size [B, C, H, W] or [B, 1, H, W]
        mask     : Tensor of size [B, C, H, W] or [B, 1, H, W]
        opposite : If true, uses log(1-x). If false, uses log(x)
        """
        if mask is not None:
            loss_fct = nn.BCELoss(reduction='sum')
        else:
            loss_fct = nn.BCELoss()

        if opposite:
            target = np.zeros(x.shape)
        else:
            target = np.ones(x.shape)

        target = Variable(torch.FloatTensor(target))
        target = target.to(self.device)

        if mask is not None:
            x = torch.where(mask.byte(), x, target)

        loss = loss_fct(x, target)

        if mask is not None:
            loss = loss / torch.sum(mask)

        return loss

    def make_prediction(self, image):
        """
        Generate class prediction for a given image
        :param image: The image used for predictions
        Returns the predicted class for each pixel of the image
        """
        raise NotImplementedError

    def _save_checkpoint(self, iteration, train_loss, valid_loss,
                         accuracy, iou, save_last=False):
        """
        Saving checkpoints. The regular checkpoint is saved only if
        the iteration is within the save period.
        If the current model is the best performing, it is saved as the
        best_checkpoint, regardless of the save period.
        :param iteration: current iteration number
        :param train_loss: current training loss
        :param valid_loss: current validation loss
        """
        save = save_last

        arch = type(self.model).__name__

        if self.best_loss > valid_loss or iteration == 0:
            self.best_loss = valid_loss
            save = True

        if self.best_iou < iou or iteration == 0:
            self.best_iou = iou
            save = True

        if self.best_accuracy < accuracy or iteration == 0:
            self.best_accuracy = accuracy
            save = True

        state = {
            'arch': arch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'state_dict_d': self.model_d.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'config': self.config,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'best_iou': self.best_iou,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'saved_iter': self.saved_iter,
            'mlflow_runid': self.mlflow_run.info.run_id
        }

        if self.best_iou == iou:
            filename_best = os.path.join(self.log_dir, 'best_iou.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best iou checkpoint.")
            self.saved_iter['best_iou'] = iteration

        if self.best_accuracy == accuracy:
            filename_best = os.path.join(self.log_dir, 'best_acc.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best accuracy checkpoint.")
            self.saved_iter['best_acc'] = iteration

        if self.best_loss == valid_loss:
            filename_best = os.path.join(self.log_dir, 'best_loss.pth')
            torch.save(state, filename_best)
            self.logger.info("Saving best loss checkpoint.")
            self.saved_iter['best_loss'] = iteration

        if save_last is True:
            filename = os.path.join(self.log_dir, 'last_checkpoint.pth')
            torch.save(state, filename)
            self.logger.info("Saving last checkpoint.")
            self.saved_iter['last_checkpt'] = iteration

        if save is True:
            with open(os.path.join(self.log_dir, 'saved_iter.txt'), 'w') as jsf:
                json.dump(self.saved_iter, jsf)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_iter = checkpoint['iteration'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is '
                'different from that of checkpoint. '
                'This may yield an exception while state_dict '
                'is being loaded.')

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model_d.load_state_dict(checkpoint['state_dict_d'])

        # DEBUG
        print('Load State dict in resume')
        print(self.model.dec_conv_00.weight.data[0][0])
        print(self.model_d.conv1.conv.weight.data[0][0])
        print('')

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
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        self.logger.info(
            "Checkpoint '{}' (iteration {}) loaded".format(resume_path,
                                                           self.start_iter))

        # Load the best metrics
        self.best_iou = checkpoint['best_iou']
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_loss = checkpoint['best_loss']

        self.saved_iter = checkpoint['saved_iter']

        self.mlflow_runid = checkpoint['mlflow_runid']

    def _adjust_lr(self, optimizer, iteration, config):
        factor = ((1 - iteration / self.config['trainer']['nb_iters'])**config['power'])
        lr = config['args']['lr'] * factor
        optimizer.param_groups[0]['lr'] = lr


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
