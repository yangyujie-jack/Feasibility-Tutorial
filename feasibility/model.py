from typing import Tuple

import casadi as ca
import numpy as np
import torch
from torch.distributions.uniform import Uniform


class Model:
    state_dim: int = 4
    action_dim: int = 2
    state_low: Tuple[float] = (-1.5, -2., 0., 0.)
    state_high: Tuple[float] = (1.5, 1., 2., np.pi)
    action_low: Tuple[float] = (-1., -np.pi / 4)
    action_high: Tuple[float] = (1., np.pi / 4)
    dt: float = 0.1


class MPCModel(Model):
    def __init__(self):
        x = ca.SX.sym('x', self.state_dim)
        u = ca.SX.sym('u', self.action_dim)
        x_prime = ca.vertcat(
            x[0] + self.dt * x[2] * ca.cos(x[3]),
            x[1] + self.dt * x[2] * ca.sin(x[3]),
            x[2] + self.dt * u[0],
            x[3] + self.dt * u[1],
        )
        self.dynamics = ca.Function('f', [x, u], [x_prime])
        self.cost = ca.Function('l', [x, u], [-x[1]])


class RLModel(Model):
    def reset(self, batch_size: int) -> torch.Tensor:
        low = torch.tensor(self.state_low, dtype=torch.float32)
        high = torch.tensor(self.state_high, dtype=torch.float32)
        return Uniform(low, high).sample((batch_size,))

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_state = torch.stack((
            state[:, 0] + self.dt * state[:, 2] * torch.cos(state[:, 3]),
            state[:, 1] + self.dt * state[:, 2] * torch.sin(state[:, 3]),
            state[:, 2] + self.dt * action[:, 0],
            state[:, 3] + self.dt * action[:, 1],
        ), dim=1)
        return next_state

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state[:, 1] - 1 - torch.abs(torch.pi / 2 - state[:, 3])

    def get_constraint(self, state: torch.Tensor) -> torch.Tensor:
        d = 0.5 - torch.norm(state[:, :2], dim=1)
        d[d <= 0] = 0.1 * d[d <= 0]
        return d

    def get_done(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, 1] > 1


def dynamics(x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
    x_prime = np.array((
        x[0] + dt * x[2] * np.cos(x[3]),
        x[1] + dt * x[2] * np.sin(x[3]),
        x[2] + dt * u[0],
        x[3] + dt * u[1],
    ))
    return x_prime


def violated(x: np.ndarray) -> bool:
    return 0.5 - np.linalg.norm(x[:2]) > 0


def terminated(x: np.ndarray) -> bool:
    return violated(x) or (np.abs(x[0]) > 1.5) or (np.abs(x[1] + 0.5) > 1.5)
