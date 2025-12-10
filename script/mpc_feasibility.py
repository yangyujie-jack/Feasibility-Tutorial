import argparse
import os
import multiprocessing as mp
from functools import partial
from typing import Sequence

import numpy as np
from feasibility.config import constraints_params
from feasibility.constraint import Constraint
from feasibility.model import MPCModel
from feasibility.path import DATA_PATH, FIGURE_PATH
from feasibility.solver import MPCSolver
from feasibility.utils import STATE_GRID, get_constraint, get_feasibility, \
    plot_feasibility, get_mpc_title


def func(state: Sequence, constraint: Constraint):
    model = MPCModel()
    solver = MPCSolver(model, constraint)
    return str(state[:2]), get_feasibility(state, solver, constraint)


def get_feasibility_grid(constraint):
    pool = mp.Pool(8)
    feas = pool.map(partial(func, constraint=constraint), STATE_GRID)
    return dict(feas)


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
        filename = f'feasibility_MPC_{args.constraint}_{str(tuple(params.values()))}'

        if args.load:
            feas = np.load(os.path.join(DATA_PATH, filename + '.npz'))
        else:
            constraint = get_constraint(args.constraint, model, **params)
            solver = MPCSolver(model, constraint)
            feas = get_feasibility_grid(constraint)
            np.savez(os.path.join(DATA_PATH, filename + '.npz'), **feas)

        plot_feasibility(feas, title, os.path.join(FIGURE_PATH, 'unicycle_' + filename + '.pdf'))
