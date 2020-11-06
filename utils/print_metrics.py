# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import torch
import argparse
from shutil import copyfile


def print_metrics(cp_path, metrics=["loss", "accuracy", "iou"], display=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(cp_path, map_location=device)

    if display is True:
        print("Loading checkpoint %s" % cp_path)

        for metric in metrics:
            print("Best %s:\t %f" % (metric, checkpoint['best_' + metric]))

    return checkpoint['best_loss'], checkpoint['best_accuracy'], checkpoint['best_iou']


def print_dir_metrics(cp_dir):
    best_loss = {}
    best_acc = {}
    best_iou = {}

    best_loss['val'] = 9999
    best_acc['val'] = 0
    best_iou['val'] = 0

    best_loss['cp'] = None
    best_acc['cp'] = None
    best_iou['cp'] = None

    for root, dirs, files in os.walk(cp_dir):
        print("Checking " + root)

        for filename in files:
            if "best_loss.pth" in filename:
                cp_path = os.path.join(root, filename)
                loss, _, _ = print_metrics(cp_path, metrics=["loss"], display=False)

                if loss < best_loss['val']:
                    best_loss['val'] = loss
                    best_loss['cp'] = cp_path

            if "best_acc.pth" in filename:
                cp_path = os.path.join(root, filename)
                _, acc, _ = print_metrics(cp_path, metrics=["accuracy"], display=False)

                if acc > best_acc['val']:
                    best_acc['val'] = acc
                    best_acc['cp'] = cp_path

            if "best_iou.pth" in filename:
                cp_path = os.path.join(root, filename)
                _, _, iou = print_metrics(cp_path, metrics=["iou"], display=False)

                if iou > best_iou['val']:
                    best_iou['val'] = iou
                    best_iou['cp'] = cp_path

    print_metrics(best_loss['cp'], metrics=["loss"])
    print_metrics(best_acc['cp'], metrics=["accuracy"])
    print_metrics(best_iou['cp'], metrics=["iou"])

    return best_loss, best_acc, best_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print metrics from a checkpoint")
    parser.add_argument('-cp', '--checkpoint', default=None, type=str,
                        help="checkpoint file path")
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help="path to dir with checkpoints")
    parser.add_argument('-b', '--best_dir', default=None, type=str,
                        help="move checkpoints to dir")
    parser.add_argument('-e', '--experiment', default=None, type=str,
                        help="experiment name")
    parser.add_argument('-t', '--trials', default=None, type=str,
                        help="trials file name")

    args = parser.parse_args()

    if args.checkpoint:
        print_metrics(args.checkpoint)
    elif args.dir:
        metrics = print_dir_metrics(args.dir)

        if args.best_dir:
            if args.experiment:
                move_dir = os.path.join(args.best_dir, args.experiment)

                os.makedirs(move_dir, exist_ok=True)

            for metric in metrics:
                new_file = os.path.join(move_dir, metric['cp'].split("/")[-1])
                copyfile(metric['cp'], new_file)

            if args.trials:
                old_trials = os.path.join(args.dir, 'trials')
                new_trials = os.path.join(move_dir, 'trials')
                copyfile(old_trials, new_trials)
