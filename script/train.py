import argparse
import time

import torch
from feasibility.adp import ADP
from feasibility.model import RLModel
from feasibility.path import PROJECT_ROOT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = RLModel()
    save_path = f'{PROJECT_ROOT}/log/' + time.strftime('%Y%m%d_%H%M%S')
    algorithm = ADP(
        model=model,
        save_path=save_path,
    )
    algorithm.train()
