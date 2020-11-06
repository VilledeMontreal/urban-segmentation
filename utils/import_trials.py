# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import json
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_data(trials_path):
    with open(trials_path, 'rb') as trials_file:
        trials = pickle.load(trials_file)

    hyperparams = list(trials.trials[0]['misc']['idxs'].keys())
    hyperparams.append('experiment')
    hyperparams.append('results')

    trials_data = pd.DataFrame(columns=hyperparams)

    for i in range(len(trials.trials)):
        values = [trials.trials[i]['misc']['vals'][hp][0] for hp in hyperparams[:-2]]
        values.append(i)
        values.append(trials.trials[i]['result']['loss'])

        trials_data.loc[i] = values

    return trials_data


def make_plots(df, config, save_path):
    cols = df.columns[:-1]

    fig, axes = plt.subplots(ncols=len(cols))

    for ax, col in zip(axes, cols):
        ax.scatter(df[col], df['results'])
        ax.title.set_text(col)
        # ax.set_ylim([-0.885, -0.87])
        try:
            ax.set_xlim(config[col])
        except KeyError:
            pass

    plt.show()

    fig.savefig(save_path)


if __name__ == "__main__":
    TRIALS_PATH = 'tri_fs_cs'
    CONFIG_PATH = 'configs/fast_scnn_cs.json'

    parser = argparse.ArgumentParser(
        description='Loading trials for analysis')
    parser.add_argument('-t', '--trials', default=TRIALS_PATH, type=str,
                        help='saved hyperopt trials path')
    parser.add_argument('-c', '--config', default=CONFIG_PATH, type=str,
                        help='config file path')

    args = parser.parse_args()

    config = json.load(open(args.config))['hyperparam']

    save_path = os.path.join('analysis', os.path.basename(args.trials) + '.png')

    trials_data = load_data(args.trials)

    make_plots(trials_data, config, save_path)
