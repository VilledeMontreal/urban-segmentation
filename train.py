# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import json
import torch
import argparse
import datetime
from utils.factories import ModelFactory, OptimizerFactory, TrainerFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Training")
    parser.add_argument('-c', '--config', default=None, type=str,
                        help="config file path (default: None)")
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help="experiment dir path (default: None)")

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
        experiment_dir = args.dir
    else:
        experiment_dir = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # Init model and optimizer from config with factories
    model = ModelFactory.get(config['model'])
    params = filter(lambda p: p.requires_grad, model.parameters())
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
            **config['trainer']['options'])

    else:
        # Init supervised trainer object from config with factory
        trainer = TrainerFactory.get(config)(
            model,
            optimizer,
            config=config,
            resume=args.resume,
            experiment_dir=experiment_dir,
            **config['trainer']['options'])

    # Run a training experiment
    trainer.train()
