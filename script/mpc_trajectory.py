import argparse
import os

import numpy as np
from feasibility.config import constraints_params
from feasibility.model import MPCModel
from feasibility.path import DATA_PATH, FIGURE_PATH
from feasibility.solver import MPCSolver
from feasibility.utils import INIT_STATE_COLOR, get_constraint, get_state_trajectory, \
    plot_trajectory, get_mpc_title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str, default='PW')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    model = MPCModel()

    constraint_params = constraints_params[args.constraint]

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)

    for params in constraint_params:
        title = get_mpc_title(args.constraint, params)
        filename = f'trajectory_MPC_{args.constraint}_{str(tuple(params.values()))}'

        if args.load:
            trajs = np.load(os.path.join(DATA_PATH, filename + '.npz'))
        else:
            constraint = get_constraint(args.constraint, model, **params)
            solver = MPCSolver(model, constraint)
            trajs = {}
            for x in INIT_STATE_COLOR.keys():
                xs = get_state_trajectory(x, solver)
                trajs[str(x[:2])] = xs
            np.savez(os.path.join(DATA_PATH, filename + '.npz'), **trajs)

        plot_trajectory(trajs, title, os.path.join(FIGURE_PATH, 'unicycle_' + filename + '.pdf'))
