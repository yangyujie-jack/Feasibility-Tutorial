from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from feasibility.config import cfg_matplotlib
from feasibility.constraint import Constraint, PWConstraint, CBFConstraint, SIConstraint, HJRConstraint
from feasibility.model import Model, dynamics, terminated, violated
from feasibility.solver import Solver
from matplotlib import rcParams


INIT_STATE_COLOR = {
    (8, 11): 'tab:blue',
    (7,  8): 'tab:green',
    (10, 5): 'tab:purple',
    (6,  3): 'tab:orange',
    (2,  1): 'tab:brown',
}

STATE_GRID = [(0.5 * i, 0.5 * j) for i in range(21) for j in range(29)]


def get_constraint(name: str, model: Model, **kwargs) -> Constraint:
    if name == 'PW':
        constraint = PWConstraint(**kwargs)
    elif name == 'CBF':
        constraint = CBFConstraint(**kwargs)
    elif name == 'SI':
        constraint = SIConstraint(
            v_max=model.state_high[1],
            a_min=model.action_low[0],
            **kwargs,
        )
    elif name == 'HJR':
        constraint = HJRConstraint(a_min=model.action_low[0])
    return constraint


def get_state_trajectory(state: Sequence, solver: Solver, step: int = 100):
    state = np.array(state)
    traj = [state]
    for _ in range(step):
        action = solver.solve(state)[1][0]
        state = dynamics(state, action)
        traj.append(state)
        if terminated(state):
            break
    return np.stack(traj)


def get_feasibility(state: Sequence, solver: Solver, constraint: Constraint, step: int = 100):
    state = np.array(state)
    traj = []
    for _ in range(step):
        states, actions = solver.solve(state)
        state = dynamics(state, actions[0])
        init_feas = constraint.initially_feasible(states)
        traj.append(init_feas)
        if terminated(state):
            break
    init_feas = traj[0]
    edls_feas = init_feas and not violated(state)
    return init_feas, edls_feas


rcParams.update({'mathtext.fontset': 'stix'})


def plot_trajectory(trajs: Dict[str, np.ndarray], title: str, save_path: str):
    plt.figure(figsize=cfg_matplotlib['fig_size'], dpi=cfg_matplotlib['dpi'])
    ax = plt.gca()

    x_lim = (-1.0, 10.5)
    y_lim = (-0.5, 14.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # plot analytical boundary
    d_red = np.linspace(0.0, x_lim[1], 100)
    u_red = np.sqrt(2 * 10 * d_red)
    ax.plot(d_red, u_red, color='#CD0F0F', linewidth=1, linestyle='--', zorder=1)
    ax.fill_between(d_red, u_red, np.ones_like(d_red) * y_lim[0], color='#FEF6F6', zorder=0)
    ax.plot([0.0, 0.0], y_lim, color='k', linewidth=1, linestyle='--', zorder=1)
    ax.fill_betweenx(y_lim, x_lim[0], 0.0, color='#DCDCDC', zorder=0)

    # plot trajectories
    for x, c in INIT_STATE_COLOR.items():
        traj = trajs[str(x)]
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5, linestyle='-', marker='o', ms=5,
                 markeredgewidth=1.5, markerfacecolor='white', zorder=2, color=c)
        plt.scatter(traj[0, 0], traj[0, 1], marker='o', color=c, zorder=2)
        if violated(traj[-1]):
            plt.scatter(traj[-1, 0], traj[-1, 1], marker='x', s=50, color='r', zorder=3)

    plt.xlabel('$d$ [m]', cfg_matplotlib['label_font'])
    plt.ylabel('$v$ [m/s]', cfg_matplotlib['label_font'])

    plt.title(title, cfg_matplotlib['label_font'])
    plt.tick_params(labelsize=cfg_matplotlib['tick_size'])

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(cfg_matplotlib['tick_label_font']) for label in labels]
    plt.tight_layout(pad=cfg_matplotlib['pad'])

    plt.savefig(save_path)


def plot_feasibility(
    feas: Dict[str, Tuple[bool, bool]],
    title: str,
    save_path: str,
):
    plt.figure(figsize=cfg_matplotlib['fig_size'], dpi=cfg_matplotlib['dpi'])
    ax = plt.gca()

    x_lim = (-0.4, 10.4)
    y_lim = (-0.5, 14.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # plot analytical boundary
    d_red = np.linspace(0.0, 10.0, 100)
    u_red = np.sqrt(2 * 10 * d_red)
    ax.plot(d_red, u_red, color='#CD0F0F', linewidth=1.5, linestyle='--', zorder=3)

    EFR = []  # endlessly feasible region
    IFR = []  # initially feasible region
    INF = []  # infeasible region
    for x_str, f in feas.items():
        x = eval(x_str)
        init_feas, edls_feas = f
        if edls_feas:
            EFR.append(x)
        elif init_feas:
            IFR.append(x)
        else:
            INF.append(x)

    if len(EFR) > 0:
        EFR = np.array(EFR)
        plt.scatter(EFR[:, 0], EFR[:, 1], marker='s', s=30, color='#EFA1A1')

    if len(IFR) > 0:
        IFR = np.array(IFR)
        plt.scatter(IFR[:, 0], IFR[:, 1], marker='D', s=20, color='#9BB0F3')

    if len(INF) > 0:
        INF = np.array(INF)
        plt.scatter(INF[:, 0], INF[:, 1], marker='D', s=20, color='#C8C8C8')

    plt.xlabel('$d$ [m]', cfg_matplotlib['label_font'])
    plt.ylabel('$v$ [m/s]', cfg_matplotlib['label_font'])

    plt.title(title, cfg_matplotlib['label_font'])
    plt.tick_params(labelsize=cfg_matplotlib['tick_size'])

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(cfg_matplotlib['tick_label_font']) for label in labels]
    plt.tight_layout(pad=cfg_matplotlib['pad'])

    plt.savefig(save_path)


def plot_legend(save_path: str, optimal: bool = True):
    plt.figure(figsize=(8, 3))

    if optimal:
        EFR_label = 'EFR of optimal policy'
        IFR_label = 'IFR'
    else:
        EFR_label = 'EFR of intermediate policy'
        IFR_label = 'IFR of intermediate policy'
    INF_label = 'Infeasible region'

    plt.scatter([0], [0], marker='s', s=30, color='#EFA1A1', label=EFR_label)
    plt.scatter([1], [0], marker='D', s=20, color='#9BB0F3', label=IFR_label)
    plt.scatter([2], [0], marker='D', s=20, color='#C8C8C8', label=INF_label)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3,
               prop=cfg_matplotlib['legend_font'])

    plt.tight_layout(pad=cfg_matplotlib['pad'])

    plt.savefig(save_path)


def get_mpc_title(constraint: str, params: dict):
    if constraint == 'PW':
        return rf'PW ($n={params["n"]}$)'
    elif constraint == 'CBF':
        return rf'CBF ($k={params["k"]}$)'
    elif constraint == 'SI':
        return rf'SI ($n={params["n"]}$, $k={params["k"]}$)'
    elif constraint == 'HJR':
        return 'HJR'


def plot_real(traj: np.ndarray, feas: Dict[str, Tuple[bool, bool]], save_path: str):
    figsize = (4.4, 4) if np.min(traj[:, 0]) < 0 else (4, 4)
    plt.figure(figsize=figsize, dpi=cfg_matplotlib['dpi'])

    EFR = []  # endlessly feasible region
    IFR = []  # initially feasible region
    INF = []  # infeasible region
    for x_str, f in feas.items():
        x = eval(x_str)
        init_feas, edls_feas = f
        if edls_feas:
            EFR.append(x)
        elif init_feas:
            IFR.append(x)
        else:
            INF.append(x)

    alpha = 0.7

    if len(EFR) > 0:
        EFR = np.array(EFR)
        plt.scatter(EFR[:, 0], EFR[:, 1], marker='s', s=30, color='#EFA1A1', alpha=alpha)

    if len(IFR) > 0:
        IFR = np.array(IFR)
        plt.scatter(IFR[:, 0], IFR[:, 1], marker='D', s=20, color='#9BB0F3', alpha=alpha)

    if len(INF) > 0:
        INF = np.array(INF)
        plt.scatter(INF[:, 0], INF[:, 1], marker='D', s=20, color='#C8C8C8', alpha=alpha)

    plt.plot(traj[:, 0], traj[:, 1], color='black', linewidth=2, zorder=4)
    plt.scatter(traj[:6, 0], traj[:6, 1], s=100, color='#9BB0F3', edgecolor='black', linewidth=2, zorder=5)
    plt.scatter(traj[6:, 0], traj[6:, 1], s=100, color='#C8C8C8', edgecolor='black', linewidth=2, zorder=5)

    violate = traj[:, 0] < 0
    plt.scatter(traj[violate, 0], traj[violate, 1], s=150, color='red', marker='x', linewidth=2, zorder=5)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)


def plot_virtual(traj: np.ndarray, real_step: int, save_path: str):
    figsize = (4.4, 4) if np.min(traj[:, 0]) < 0 else (4, 4)
    plt.figure(figsize=figsize, dpi=cfg_matplotlib['dpi'])

    # plot analytical boundary
    d_red = np.linspace(0.0, 10.0, 100)
    u_red = np.sqrt(2 * 10 * d_red)
    plt.plot(d_red, u_red, color='#EA700E', linewidth=3, zorder=3)
    plt.fill_between(d_red, u_red, u_red[-1], color='#EA700E', alpha=0.3, zorder=3)

    plt.plot(traj[:real_step + 1, 0], traj[:real_step + 1, 1], color='black', linewidth=2, zorder=4)
    plt.plot(traj[real_step:, 0], traj[real_step:, 1], color='black', linewidth=2, linestyle='--', zorder=4)
    plt.scatter(traj[0, 0], traj[0, 1], s=100, color='#EFA1A1', edgecolor='black', linewidth=2, zorder=5)
    plt.scatter(traj[1:, 0], traj[1:, 1], s=100, color='#EFA1A1', edgecolor='black', linewidth=2, zorder=5)

    violate = traj[:, 1] > np.sqrt(2 * 10 * np.maximum(traj[:, 0], 0))
    plt.scatter(traj[violate, 0], traj[violate, 1], s=150, color='red', marker='x', linewidth=2, zorder=5)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
