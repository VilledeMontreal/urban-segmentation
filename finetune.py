# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import json
import torch
import pickle
import argparse
import datetime
from hyperopt import tpe, hp, fmin, Trials
from utils.factories import ModelFactory, OptimizerFactory, TrainerFactory


def _train(hyperparams):
    # Path to subdir for single run
    dir_path = os.path.join(config['trainer']['log_dir'],
                            config['name'], main_experiment_dir)
    try:
        run = sum(os.path.isdir(os.path.join(dir_path, i)) for i in os.listdir(dir_path))
    except FileNotFoundError:
        run = 0

    experiment_dir = main_experiment_dir + '/' + str(run)

    # Init model and optimizer with factories
    model = ModelFactory.get(config['model'])
    params = filter(lambda p: p.requires_grad, model.parameters())
    config['optimizer']['args']['lr'] = hyperparams['lr']
    optimizer = OptimizerFactory.get(config['optimizer'], params)

    # Check if semi-supervised run
    if config['semi'] is True:
        # Init model_d and optimizer_d from config with factories
        model_d = ModelFactory.get(config['model_d'])
        params_d = filter(lambda p: p.requires_grad, model_d.parameters())
        optimizer_d = OptimizerFactory.get(config['optimizer_d'], params_d)

        # Init semi-supervised trainer object from config with factory
        trainer = TrainerFactory.get(config)(
            model,
            model_d,
            optimizer,
            optimizer_d,
            config=config,
            resume=args.resume,
            experiment_dir=experiment_dir,
            hyperparams=hyperparams,
            **config['trainer']['options'])

    else:
        # Init supervised trainer object from config with factory
        trainer = TrainerFactory.get(config)(
            model,
            optimizer,
            config=config,
            resume=args.resume,
            experiment_dir=experiment_dir,
            hyperparams=hyperparams,
            **config['trainer']['options'])

    # Run single training experiment
    iou = trainer.train()

    return -iou


def _create_space(config):
    # Creating hyperparam space from config for hyperopt
    space = {}
    for key in config:
        space[key] = hp.uniform(key, config[key][0], config[key][1])
    return space


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Training")
    parser.add_argument('-c', '--config', default=None, type=str,
                        help="config file path (default: None)")
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help="experiment dir path (default: None)")
    parser.add_argument('-t', '--trials', default=None, type=str,
                        help="saved hyperopt trials path (default: None)")

    args = parser.parse_args()

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.deterministic = True

    # Check if Colab run
    COLAB = os.path.exists("/content/gdrive")

    if args.config:
        # Load config file
        config = json.load(open(args.config))
    elif args.resume:
        # Load config file from checkpoint
        config = torch.load(args.resume, map_location=device)['config']

    # Change log dir if colab run
    if COLAB is True:
        config['trainer']['log_dir'] = "/content/gdrive/My Drive/colab_saves/logs/"

    # Set experiment dir to current time if none provided
    if args.dir:
        main_experiment_dir = args.dir
    else:
        main_experiment_dir = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # Init Hyperopt settings for hyperparameter optimization
    tpe_algo = tpe.suggest
    space = _create_space(config['hyperparam'])
    trials_savepath = os.path.join(config['trainer']['log_dir'],
                                   config['name'], main_experiment_dir, 'trials')

    # Load or create Trials object
    if args.trials:
        with open(args.trials, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        trials = Trials()

    starting_eval = len(trials)

    for i in range(starting_eval, config['trainer']['max_evals']):
        if i > starting_eval:
            # Load previous trials
            with open(trials_savepath, 'rb') as trials_file:
                trials = pickle.load(trials_file)

        # Init and run a single training experiment
        tpe_best = fmin(fn=_train,
                        space=space,
                        algo=tpe_algo,
                        max_evals=i + 1,
                        trials=trials)

        # Save the trial file after a single run
        with open(trials_savepath, 'wb') as trials_file:
            pickle.dump(trials, trials_file)
