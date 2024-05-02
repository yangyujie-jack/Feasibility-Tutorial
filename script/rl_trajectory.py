import argparse
import os

import numpy as np
import torch
from feasibility.model import RLModel
from feasibility.network import IHPolicy
from feasibility.path import DATA_PATH, FIGURE_PATH, LOG_PATH
from feasibility.solver import RLSolver
from feasibility.utils import INIT_STATE_COLOR, get_state_trajectory, plot_trajectory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str, default='SI')
    parser.add_argument('--log_dir', type=str, default='20240409_012948')
    args = parser.parse_args()

    model = RLModel()

    policy = IHPolicy(
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        action_low=model.action_low,
        action_high=model.action_high,
    )

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)

    log_dir = os.path.join(LOG_PATH, args.constraint, args.log_dir)
    for f in os.listdir(log_dir):
        if not f.startswith('ckpts'):
            continue

        ckpt_iter = f[6:-3]
        load_path = os.path.join(log_dir, f)
        policy.load_state_dict(torch.load(load_path)['policy'])

        solver = RLSolver(policy)

        trajs = {}
        for state in INIT_STATE_COLOR.keys():
            trajs[str(state)] = get_state_trajectory(state, solver)

        filename = f'trajectory_RL_{args.constraint}_{ckpt_iter}'
        np.savez(os.path.join(DATA_PATH, filename + '.npz'), **trajs)

        plot_trajectory(trajs, args.constraint + f' (iter {ckpt_iter})',
                        os.path.join(FIGURE_PATH, filename + '.png'))
