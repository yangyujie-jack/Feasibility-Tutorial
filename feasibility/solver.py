from typing import Tuple

import casadi as ca
import numpy as np
import torch
from feasibility.constraint import Constraint
from feasibility.model import MPCModel, RLModel, dynamics
from feasibility.network import IHPolicy


EPSILON = 1e-4


class Solver:
    name: str

    def solve(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Input initial state and output state and action trajectories.
        '''
        raise NotImplementedError


class MPCSolver(Solver):
    '''
    NLP solver for nonlinear model predictive control with Casadi.
    '''

    name: str = 'MPC'

    def __init__(self, model: MPCModel, constraint: Constraint, pre_horizon: int = 10):
        # self.cfg = cfg
        self.model = model
        self.pre_horizon = pre_horizon

        # create empty NLP
        w = []         # optimization variables, including states and actions
        self.lbw = []  # lower bounds on `w`
        self.ubw = []  # upper bounds on `w`
        g = []         # constraint functions
        self.lbg = []  # lower bounds on `g`
        self.ubg = []  # upper bounds on `g`
        J = 0.0        # cost function

        # initial state
        x = ca.MX.sym('x0', model.state_dim)
        w += [x]
        self.lbw += [-ca.inf] * model.state_dim
        self.ubw += [ca.inf] * model.state_dim

        for k in range(pre_horizon):
            # action
            u = ca.MX.sym('u' + str(k), model.action_dim)
            w += [u]
            self.lbw += list(model.action_low)
            self.ubw += list(model.action_high)

            # next state
            x_prime = ca.MX.sym('x' + str(k + 1), model.state_dim)
            w += [x_prime]
            self.lbw += [-ca.inf] * model.state_dim
            self.ubw += [ca.inf] * model.state_dim

            # dynamics constraint
            g += [model.dynamics(x, u) - x_prime]
            self.lbg += [0.0] * model.state_dim
            self.ubg += [0.0] * model.state_dim

            if k < constraint.step:
                g += [constraint.ca_constraint(x, x_prime)]
                self.lbg += [-ca.inf]
                self.ubg += [-EPSILON]

            # cost function
            J += model.cost(u)

            # update state
            x = x_prime

        # create NLP solver
        nlp = dict(f=J, g=ca.vertcat(*g), x=ca.vertcat(*w))
        sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.S = ca.nlpsol('S', 'ipopt', nlp, sol_dic)

    def solve(self, x: np.ndarray):
        '''
        Solver of nonlinear MPC

        Parameters
        ----------
        x: np.ndarray
            input state for MPC.

        Returns
        ----------
        state: np.ndarray
            state trajectory of MPC in the whole predict horizon.
        action: np.ndarray
            action trajectory of MPC in the whole predict horizon.
        '''

        self.lbw[:len(x)] = list(x)
        self.ubw[:len(x)] = list(x)

        # solve NLP
        result = self.S(lbx=self.lbw, ubx=self.ubw, x0=0, lbg=self.lbg, ubg=self.ubg)
        solution = np.array(result['x'])

        # get trajectory
        state = np.zeros([self.pre_horizon + 1, self.model.state_dim])
        action = np.zeros([self.pre_horizon, self.model.action_dim])
        state[0] = x
        N, M = self.model.state_dim, self.model.action_dim
        for i in range(self.pre_horizon):
            action[i] = solution[(N + M) * i + N: (N + M) * (i + 1), 0]
            state[i + 1] = dynamics(state[i], action[i])

        return state, action


class RLSolver(Solver):
    name: str = 'RL'

    def __init__(self, policy: IHPolicy, forward_step: int = 10):
        self.policy = policy
        self.forward_step = forward_step

    def solve(self, state: np.ndarray):
        state_traj = [state]
        action_traj = []
        for _ in range(self.forward_step):
            state = torch.from_numpy(state).float()
            with torch.no_grad():
                action = self.policy(state).numpy()
            state = dynamics(state, action)
            state_traj.append(state)
            action_traj.append(action)
        return np.stack(state_traj), np.stack(action_traj)
