import argparse
import os
import multiprocessing as mp
from functools import partial
from typing import Sequence

import numpy as np
import torch
from feasibility.path import DATA_PATH, FIGURE_PATH, LOG_PATH
from feasibility.solver import RLSolver
from feasibility.model import RLModel
from feasibility.network import IHPolicy, IHValue
from feasibility.constraint import Constraint
from feasibility.utils import STATE_GRID, get_constraint, get_feasibility, plot_feasibility


def func(state: Sequence, policy: IHPolicy, constraint: Constraint):
    solver = RLSolver(policy)
    return str(state), get_feasibility(state, solver, constraint)


def get_feasibility_grid(policy: IHPolicy, constraint: Constraint):
    pool = mp.Pool(8)
    feas = pool.map(partial(func, policy=policy, constraint=constraint), STATE_GRID)
    return dict(feas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='20251207_114428')
    args = parser.parse_args()

    model = RLModel()

    policy = IHPolicy(
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        action_low=model.action_low,
        action_high=model.action_high,
    )
    feasibility = IHValue(
        state_dim=model.state_dim
    )

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)

    log_dir = os.path.join(LOG_PATH, args.log_dir)
    for f in os.listdir(log_dir):
        if not f.startswith('ckpts'):
            continue

        ckpt_iter = f[6:-3]
        load_path = os.path.join(log_dir, f)
        state_dict = torch.load(load_path)
        policy.load_state_dict(state_dict['policy'])
        feasibility.load_state_dict(state_dict['feasibility'])

        constraint = get_constraint('HJR', model, feasibility=feasibility)
        feas = get_feasibility_grid(policy, constraint)

        np.savez(os.path.join(DATA_PATH, 'feasibility_RL.npz'), **feas)

        plot_feasibility(feas, f'HJR (iter {ckpt_iter})',
                         os.path.join(FIGURE_PATH, 'unicycle_feasibility_RL.pdf'))
