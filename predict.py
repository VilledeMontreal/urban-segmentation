# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import torch
import argparse
import datetime
from utils.factories import PredictFactory, ModelFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation Training')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help='experiment dir path (default: None)')

    args = parser.parse_args()

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.checkpoint:
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Get config from checkpoint
        config = checkpoint['config']

        # If you want to change the items in the checkpoint config,
        # e.g. path to the datasets, just catch it and modify the dict.
        # config['data']['dataset']['data_dir'] = '../CGMU_L'

        # Init model from checkpoint config with factory
        model = ModelFactory.get(config['model'])

        # Load model weights
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        torch.no_grad()

        # Set experiment dir to current time if none provided
        if args.dir:
            experiment_dir = args.dir
        else:
            experiment_dir = datetime.datetime.now().strftime("%m%d_%H%M%S")

        # Init predictor object from config with factory
        predictor = PredictFactory.get(config)(
            model,
            config=config,
            experiment_dir=experiment_dir,
            **config['predictor']['options']
        )

        # Make predictions
        predictor.predict()

    else:
        print('Use --checkpoint or -c to give the path to the checkpoint to load')
