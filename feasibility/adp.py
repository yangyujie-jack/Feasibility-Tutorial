import os
from copy import deepcopy

import torch
from torch.optim import Adam
from feasibility.model import RLModel
from feasibility.network import IHValue, IHPolicy
from torch.utils.tensorboard import SummaryWriter


class ADP:
    def __init__(
        self,
        model: RLModel,
        save_path: str,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        reward_scale: float = 1.,
        penalty: float = 100.,
        max_iter: int = 40000,
        log_every: int = 100,
    ):
        self.model = model
        self.save_path = save_path
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.penalty = penalty
        self.max_iter = max_iter
        self.log_every = log_every

        self.value = IHValue(
            state_dim=model.state_dim,
        )
        self.target_value = deepcopy(self.value)
        self.target_value.requires_grad_(False)
        self.feasibility = IHValue(
            state_dim=model.state_dim,
        )
        self.target_feasibility = deepcopy(self.feasibility)
        self.target_feasibility.requires_grad_(False)
        self.policy = IHPolicy(
            state_dim=model.state_dim,
            action_dim=model.action_dim,
            action_low=model.action_low,
            action_high=model.action_high,
        )

        self.value_optimizer = Adam(self.value.parameters(), lr=lr)
        self.feasibility_optimizer = Adam(self.feasibility.parameters(), lr=lr)
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
            constraint = self.model.get_constraint(state)
            next_state = self.model.get_next_state(state, action)
            done = self.model.get_done(next_state)

            # update value
            v_pred = self.value(state)
            v_targ = reward + ~done * self.gamma * self.target_value(next_state)
            v_loss = ((v_pred - v_targ.detach()) ** 2).mean()
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()

            # update feasibility
            f_pred = self.feasibility(state)
            f_targ = (1 - self.gamma) * constraint + ~done * self.gamma * \
                torch.max(constraint, self.target_feasibility(next_state))
            f_loss = ((f_pred - f_targ.detach()) ** 2).mean()
            self.feasibility_optimizer.zero_grad()
            f_loss.backward()
            self.feasibility_optimizer.step()

            # update policy
            ret = reward + ~done * self.gamma * self.value(next_state)
            cstr = ~done * self.feasibility(next_state)
            feas = cstr < 0.
            policy_loss = (feas * -ret + ~feas * self.penalty * cstr).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # update target networks
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            for target_param, param in zip(self.target_feasibility.parameters(), self.feasibility.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            if (i + 1) % self.log_every == 0:
                print('Iteration', i + 1)
                writer.add_scalar('train/value loss', v_loss.item(), i + 1)
                writer.add_scalar('train/feasibility loss', f_loss.item(), i + 1)
                writer.add_scalar('train/policy loss', policy_loss.item(), i + 1)
                writer.add_scalar('train/feasible ratio', feas.float().mean().item(), i + 1)

        torch.save(
            {
                'value': self.value.state_dict(),
                'feasibility': self.feasibility.state_dict(),
                'policy': self.policy.state_dict(),
            }, 
            f'{self.save_path}/ckpts_{self.max_iter}.pt'
        )
        print(f'Network params saved at iter {self.max_iter}!')
