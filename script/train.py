import argparse
import time

import torch
from feasibility.adp import ADP
from feasibility.model import RLModel
from feasibility.path import PROJECT_ROOT
from feasibility.utils import get_constraint


config = {
    'PW': {
        'reward_scale': 0.01,
        'penalty': 0.2,
        'save_at': (10, 50, 100, 10000),
    },
    'CBF': {
        'reward_scale': 0.01,
        'penalty': 0.05,
        'save_at': (10, 50, 100, 10000),
    },
    'SI': {
        'reward_scale': 1e-4,
        'penalty': 1e-3,
        'save_at': (10, 50, 100, 10000),
    },
    'HJR': {
        'reward_scale': 1e-4,
        'penalty': 0.02,
        'save_at': (10, 50, 100, 10000),
    }
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str, default='SI')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = RLModel()
    constraint = get_constraint(args.constraint, model)
    save_path = f'{PROJECT_ROOT}/log/' + args.constraint + '/' + time.strftime('%Y%m%d_%H%M%S')
    algorithm = ADP(
        model=model,
        constraint=constraint,
        save_path=save_path,
        **config[args.constraint],
    )
    algorithm.train()
