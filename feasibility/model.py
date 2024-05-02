from typing import Tuple

import casadi as ca
import numpy as np
import torch
from torch.distributions.uniform import Uniform


class Model:
    state_dim: int = 2
    action_dim: int = 1
    state_low: Tuple[float] = (0.0, 0.0)
    state_high: Tuple[float] = (10.5, 14.5)
    action_low: Tuple[float] = (-10.0,)
    action_high: Tuple[float] = (0.0,)
    dt: float = 0.1


class MPCModel(Model):
    def __init__(self):
        x = ca.SX.sym('x', self.state_dim)
        u = ca.SX.sym('u', self.action_dim)
        x_prime = ca.vertcat(
            x[0] - self.dt * x[1],
            x[1] + self.dt * u[0],
        )
        self.dynamics = ca.Function('f', [x, u], [x_prime])
        self.cost = ca.Function('l', [u], [u[0] ** 2])


class RLModel(Model):
    def reset(self, batch_size: int) -> torch.Tensor:
        low = torch.tensor(self.state_low, dtype=torch.float32)
        high = torch.tensor(self.state_high, dtype=torch.float32)
        return Uniform(low, high).sample((batch_size,))

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_state = torch.stack((
            state[:, 0] - self.dt * state[:, 1],
            state[:, 1] + self.dt * action[:, 0],
        ), dim=1)
        return next_state

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return -action[:, 0] ** 2

    def get_done(self, state: torch.Tensor) -> torch.Tensor:
        return (state[:, 0] < 0.0) | (state[:, 1] < 0.0)


def dynamics(x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
    x_prime = np.array((
        x[0] - dt * x[1],
        x[1] + dt * u[0],
    ))
    return x_prime


def violated(x: np.ndarray) -> bool:
    return x[0] < 0


def terminated(x: np.ndarray) -> bool:
    return violated(x) or x[1] <= 0
