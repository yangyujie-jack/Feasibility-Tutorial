from typing import Sequence

import torch
from torch import nn


class IHValue(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class IHPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.low = torch.tensor(action_low, dtype=torch.float32)
        self.high = torch.tensor(action_high, dtype=torch.float32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * (self.high - self.low) / 2 + (self.high + self.low) / 2
