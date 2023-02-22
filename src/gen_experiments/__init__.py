from mitosis import Parameter

from . import sho
from . import lorenz
from . import nonlinear_pendulum
from . import cubic_oscillator
from . import vanderpol
from . import hopf
from . import odes
from . import lorenz_missing
from . import wrapper

experiments = {
    "sho": (sho, None),
    "lorenz": (lorenz, None),
    "lorenz_2d": (lorenz_missing, None),
    "pendulum": (nonlinear_pendulum, None),
    "cubic_ho": (cubic_oscillator, None),
    "vdp": (vanderpol, None),
    "hopf": (hopf, None),
    "duff": (odes, "duff"),
    "lv": (odes, "lv"),
    "ross": (odes, "ross"),
    "wrapper": (wrapper, None),
}
ex_name = type("identidict", (), {"__getitem__": lambda self, key: key})()


def lookup_params(params: list[str]) -> list[Parameter]:
    resolved_params = []
    for param in params:
        p_name, p_id = param.split("=")
        choices = globals()[p_name]
        resolved_params.append(Parameter(p_id, p_name, choices[p_id]))
    return resolved_params


def lookup_grid_params(params: list[str]) -> list[Parameter]:
    if not params:
        return None
    grid_params = []
    grid_vals = []
    grid_decisions = []
    other_params = {}
    for param in params:
        p_name, p_id = param.split("=")
        if p_name != "grid":
            other_params[p_name] = globals()[p_name][p_id]
        else:
            vals = p_id.split(";")
            grid_params.append(vals[0])
            grid_vals.append([float(val) for val in vals[1].split(",")])
            grid_decisions.append(vals[2])
    return [
        Parameter("combo", "grid_params", grid_params),
        Parameter("combo", "grid_vals", grid_vals),
        Parameter("combo", "grid_decisions", grid_decisions),
        Parameter("combo", "other_params", other_params),
    ]


sim_params = {
    "test": {"n_trajectories": 2},
    "test2": {"n_trajectories": 2, "noise_stdev": 0.4},
    "med-noise": {"n_trajectories": 2, "noise_stdev": 0.8},
    "hi-noise": {"n_trajectories": 2, "noise_stdev": 2},
}
diff_params = {
    "test": {"diffcls": "FiniteDifference"},
    "test2": {"diffcls": "SmoothedFiniteDifference"},
    "sfd-nox": {"diffcls": "SmoothedFiniteDifference", "save_smooth": False},
    "kalman": {"diffcls": "sindy", "kind": "kalman", "alpha": 0.000055},
}
feat_params = {
    "test": {"featcls": "Polynomial"},
    "test2": {"featcls": "Fourier"},
    "test3": {"featcls": "Polynomial", "degree": 3},
}
opt_params = {"test": {"optcls": "STLSQ"}}

# Grid search parameters
metrics = {"test": ["coeff_f1", "coeff_mae"]}
other_params = {
    "test": {
        "sim_params": sim_params["test"],
        "diff_params": diff_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }
}
grid_params = {"test": ["sim_params.t_end"]}
grid_vals = {"test":[[5,10,15,20]]}
grid_decisions = {"test": ["plot"]}
