constraints_params = {
    'PW': (
        {'n': 2},
        {'n': 4},
        {'n': 6},
        {'n': 10},
    ),
    'CBF': (
        {'k': 0.5},
        {'k': 0.2},
        {'k': 0.1},
        {'k': 0.05},
    ),
    'SI': (
        {'n': 0.5, 'k': 0.23},
        {'n': 0.5, 'k': 0.5},
        {'n': 1, 'k': 1},  # SI condition violated
        {'n': 2, 'k': 5},  # SI condition violated
    ),
    'HJR': (
        {},
    )
}

default_rl_config = dict(
    solver_name = 'rl',
    state_dim = 2,
    action_dim = 1,
    action_low = -10.0,
    action_high = 0.0,
    Ts = 0.1,  # control period, [s]
    save = False,  # save trajectory
)

cfg_matplotlib = dict()

cfg_matplotlib["fig_size"] = (4, 4)
cfg_matplotlib["dpi"] = 300
cfg_matplotlib["pad"] = 0.5

cfg_matplotlib["tick_size"] = 10
cfg_matplotlib["tick_label_font"] = "Times New Roman"
cfg_matplotlib["legend_font"] = {
    "family": "Times New Roman",
    "size": "10",
    "weight": "normal",
}
cfg_matplotlib["label_font"] = {
    "family": "Times New Roman",
    "size": "16",
    "weight": "normal",
}
