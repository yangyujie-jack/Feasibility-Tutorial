import warnings
from typing import Union

import numpy as np
import torch
from casadi import MX, if_else


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
        return -x_prime[0]

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return (-x[:self.step + 1, 0] <= 0).all()

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return -x_prime[..., 0]


class CBFConstraint(Constraint):
    name: str = 'CBF'
    step: int = 1

    def __init__(self, k: float = 0.05, alpha: float = 0.1):
        self.k = k
        self.alpha = alpha

    def ca_function(self, x: MX) -> MX:
        # x.shape = (state_dim, 1)
        return -x[0] + self.k * x[1] ** 2

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        return self.ca_function(x_prime) - (1 - self.alpha) * self.ca_function(x)

    def function(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, float, torch.Tensor]:
        # x.shape = (..., state_dim)
        return -x[..., 0] + self.k * x[..., 1] ** 2

    def np_constraint(self, x: np.ndarray) -> Union[np.ndarray, float]:
        # x.shape = (..., traj_length, state_dim)
        return self.function(x[..., 1, :]) - (1 - self.alpha) * self.function(x[..., 0, :])

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return self.function(x[0]) <= 0 and self.np_constraint(x) <= 0

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return self.function(x_prime) - (1 - self.alpha) * self.function(x)


class SIConstraint(Constraint):
    """
    Safet index from:
    Zhao, W., He, T., & Liu, C. (2021, June). 
    Model-free safe control for zero-violation reinforcement learning. 
    In 5th Annual Conference on Robot Learning.
    """

    name: str = 'SI'
    step: int = 1

    def __init__(
        self, 
        sigma: float = 0.12,
        d_min: float = 0.0,
        n: float = 0.5,
        k: float = 0.23,
        eta: float = 0.0,
        v_max: float = 1.0,
        a_min: float = -1.0,
        eps: float = 1e-8,
    ):
        # check Equation (3)
        if n * (sigma + d_min ** n + k * v_max) ** ((n - 1) / n) / k > -a_min / v_max:
            warnings.warn('SI forward invariant condition violated!') 

        self.sigma = sigma
        self.d_min = d_min
        self.n = n
        self.k = k
        self.eta = eta
        self.v_max = v_max
        self.a_min = a_min
        self.eps = eps

    def ca_function(self, x: MX) -> MX:
        # x.shape = (state_dim, 1)
        return self.sigma + self.d_min ** self.n - \
            if_else(x[0] > self.eps, x[0], self.eps) ** self.n + self.k * x[1]

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        tmp = self.ca_function(x) - self.eta
        return self.ca_function(x_prime) - if_else(tmp > 0, tmp, 0)

    def np_function(self, x: np.ndarray) -> Union[np.ndarray, float]:
        # x.shape = (..., state_dim)
        return self.sigma - np.maximum(x[..., 0], self.eps) ** self.n + self.k * x[..., 1]

    def np_constraint(self, x: np.ndarray) -> Union[np.ndarray, float]:
        # x.shape = (..., traj_length, state_dim)
        return self.np_function(x[..., 1, :]) - np.maximum(self.np_function(x[..., 0, :]) - self.eta, 0)

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return self.np_function(x[0]) <= 0 and self.np_constraint(x) <= 0

    def torch_function(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return self.sigma - torch.clamp_min(x[..., 0], self.eps) ** self.n + self.k * x[..., 1]

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return self.torch_function(x_prime) - torch.clamp_min(self.torch_function(x) - self.eta, 0)


class HJRConstraint(Constraint):
    name: str = 'HJR'
    step: int = 4

    def __init__(self, a_min: float = -1.0):
        self.a_min = a_min

    def ca_constraint(self, x: MX, x_prime: MX) -> MX:
        # x.shape = (state_dim, 1)
        return -x_prime[0] - 1 / (2 * self.a_min) * x_prime[1] ** 2

    def function(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, float, torch.Tensor]:
        # x.shape = (..., state_dim)
        return -x[..., 0] - 1 / (2 * self.a_min) * x[..., 1] ** 2

    def initially_feasible(self, x: np.ndarray) -> bool:
        # x.shape = (traj_length, state_dim)
        return self.function(x[0]) <= 0 and self.function(x[1]) <= 0

    def torch_constraint(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., state_dim)
        return self.function(x_prime)
