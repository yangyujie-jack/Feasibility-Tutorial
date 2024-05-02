import os
from copy import deepcopy
from typing import Optional, Sequence

import torch
from torch.optim import Adam
from feasibility.constraint import Constraint
from feasibility.model import RLModel
from feasibility.network import IHValue, IHPolicy
from torch.utils.tensorboard import SummaryWriter


EPSILON = 1e-4


class ADP:
    def __init__(
        self,
        model: RLModel,
        constraint: Constraint,
        save_path: str,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        reward_scale: float = 1.0,
        penalty: float = 1.0,
        max_iter: int = 10000,
        save_at: Optional[Sequence] = None,
    ):
        self.model = model
        self.constraint = constraint
        self.save_path = save_path
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.penalty = penalty
        self.max_iter = max_iter
        self.save_at = save_at

        self.value = IHValue(
            state_dim=model.state_dim,
        )
        self.target_value = deepcopy(self.value)
        self.target_value.eval()
        self.policy = IHPolicy(
            state_dim=model.state_dim,
            action_dim=model.action_dim,
            action_low=model.action_low,
            action_high=model.action_high,
        )

        self.value_optimizer = Adam(self.value.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def train(self):
        os.makedirs(self.save_path, exist_ok=True)
        writer = SummaryWriter(self.save_path)

        for i in range(self.max_iter):
            state = self.model.reset(self.batch_size)

            # rollout
            action = self.policy(state)
            reward = self.model.get_reward(state, action)
            reward = self.reward_scale * reward
            next_state = self.model.get_next_state(state, action)
            done = self.model.get_done(next_state)

            # update value
            v_pred = self.value(state)
            v_targ = reward + ~done * self.gamma * self.target_value(next_state)
            v_loss = ((v_pred - v_targ.detach()) ** 2).mean()
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()

            # update policy
            ret = reward + ~done * self.gamma * self.value(state)

            if self.constraint.name == 'PW':
                x = next_state
                cstrs = [self.constraint.torch_constraint(x, x)]
                for _ in range(self.constraint.step - 1):
                    u = self.policy(x).detach()
                    x = self.model.get_next_state(x, u)
                    cstrs.append(self.constraint.torch_constraint(x, x))
                cstr = torch.max(torch.stack(cstrs, dim=1), dim=1).values
            else:
                cstr = self.constraint.torch_constraint(state, next_state)

            feas = cstr <= -EPSILON
            feas_loss = ((-ret - self.penalty * torch.log(torch.clamp_min(-cstr, EPSILON))) * feas).mean()
            infe_loss = (cstr * ~feas).mean()
            policy_loss = feas_loss + infe_loss
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # update target value
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            writer.add_scalar('train/policy loss', policy_loss.item(), i + 1)
            writer.add_scalar('train/value loss', v_loss.item(), i + 1)
            writer.add_scalar('train/feasible ratio', feas.float().mean().item(), i + 1)

            if i + 1 in self.save_at:
                torch.save(
                    {
                        'policy': self.policy.state_dict(),
                        'valie': self.value.state_dict(),
                    }, 
                    f'{self.save_path}/ckpts_{i + 1}.pt'
                )
                print(f'Network params saved at iter {i + 1}!')
