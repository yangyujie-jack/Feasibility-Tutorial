from typing import Union

import numpy as np
import torch
import casadi as ca
from casadi import MX


EPSILON = 1e-8


class Constraint:
    name: str
    step: int

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        raise NotImplementedError

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        raise NotImplementedError

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        raise NotImplementedError


class PWConstraint(Constraint):
    name: str = 'PW'

    def __init__(self, n: float = 10):
        self.step = n

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        return 0.25 - (x_prime[0] ** 2 + x_prime[1] ** 2)

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return (0.5 - np.linalg.norm(x[:self.step + 1, :2], axis=1) <= 0).all()

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return 0.5 - torch.norm(x_prime[..., :2], dim=-1)


class CBFConstraint(Constraint):
    name: str = 'CBF'
    step: int = 1

    def __init__(self, k: float = 0.05, alpha: float = 0.1):
        self.k = k
        self.alpha = alpha

    def ca_function(self, x: MX) -> MX:
        # x.shape = (state_dim, 1)
        d = ca.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = ca.atan2(x[1], ca.if_else(ca.fabs(x[0]) > EPSILON, x[0], EPSILON))
        return 0.7 - ca.if_else(d > EPSILON, d, EPSILON) - self.k * x[2] * ca.cos(x[3] - phi)

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        return self.ca_function(x_prime) - (1 - self.alpha) * self.ca_function(x)

    def function(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, float, torch.Tensor]:
        # x.shape = (..., state_dim)
        return 0.7 - np.linalg.norm(x[..., :2]) - \
            self.k * x[..., 2] * np.cos(x[..., 3] - np.arctan2(x[..., 1], x[..., 0]))

    def np_constraint(self, x: np.ndarray) -> Union[np.ndarray, float]:
        # x.shape = (..., traj_length, state_dim)
        return self.function(x[..., 1, :]) - (1 - self.alpha) * self.function(x[..., 0, :])

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return self.function(x[0]) <= 0 and self.np_constraint(x) <= 0

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return self.function(x_prime) - (1 - self.alpha) * self.function(x)


class HJRConstraint(Constraint):
    name: str = 'HJR'
    step: int = 1

    def __init__(self, feasibility: torch.nn.Module):
        self.feasibility = feasibility

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        with torch.no_grad():
            f = self.feasibility(torch.as_tensor(x[:2]).float()).numpy()
        return f[0] <= 0.05 and f[1] <= 0.05
