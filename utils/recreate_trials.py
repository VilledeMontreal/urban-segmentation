# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import pickle
from functools import partial
from hyperopt import tpe, hp, fmin, Trials

result_list = {}
result_list['lr'] = (
    [0.00142072954901, -0.42065678023921],
    [0.00141600471635, -0.39946523159365],
    [0.00066498850972, -0.436678247860291]
)


def load_trials(path):
    with open(path, 'rb') as trials_file:
        trials = pickle.load(trials_file)

    return trials


def save_trials(path, trials):
    with open(path, 'wb') as trials_file:
        pickle.dump(trials, trials_file)


def create_space(result_list, i):
    space = {}
    for key in result_list:
        space[key] = hp.uniform(key, result_list[key][i][0], result_list[key][i][0])

    return space


def objective(i_res, x):
    # Must be changed for multi-output
    print(i_res[1]['lr'][i_res[0]][1])
    return i_res[1]['lr'][i_res[0]][1]


def create_trial(result_list):
    trials = Trials()
    tpe_algo = tpe.suggest

    for i in range(len(result_list[next(iter(result_list))])):
        if i == 0:
            trials = Trials()
        else:
            trials = load_trials('trials')

        space = create_space(result_list, i)

        fmin(fn=partial(objective, (i, result_list)),
             space=space,
             algo=tpe_algo,
             max_evals=i + 1,
             trials=trials)

        save_trials('trials', trials)


if __name__ == "__main__":

    # create_trial(result_list)
    trials = load_trials('trials')
    print(len(trials))
