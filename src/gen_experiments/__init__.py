from collections import namedtuple

import numpy as np
import pysindy as ps

from mitosis import Parameter

from . import sho
from . import lorenz
from . import nonlinear_pendulum
from . import cubic_oscillator
from . import vanderpol
from . import hopf
from . import odes
from . import lorenz_missing
from . import gridsearch

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
    "wrapper": (gridsearch, None),
}
ex_name = type("identidict", (), {"__getitem__": lambda self, key: key})()


def lookup_params(params: list[str]) -> list[Parameter]:
    resolved_params = []
    for param in params:
        p_name, p_id = param.split("=")
        choice = globals()[p_name][p_id]
        try:
            vals = choice.vals
            modules = choice.modules
        except AttributeError:
            vals = choice
            modules = []
        resolved_params.append(Parameter(p_id, p_name, vals, modules))
    return resolved_params


ParamDetails = namedtuple("ParamDetails", ["vals", "modules"])
sim_params = {
    "test": {"n_trajectories": 2},
    "test2": {"n_trajectories": 2, "noise_stdev": 0.4},
    "med-noise": {"n_trajectories": 2, "noise_stdev": 0.8},
    "hi-noise": {"n_trajectories": 2, "noise_stdev": 2},
}
diff_params = {
    "test": {"diffcls": "FiniteDifference"},
    "test2": {"diffcls": "SmoothedFiniteDifference"},
    "tv": {"diffcls": "sindy", "kind": "trend_filtered"},
    "sfd-nox": {"diffcls": "SmoothedFiniteDifference", "save_smooth": False},
    "kalman": {"diffcls": "sindy", "kind": "kalman", "alpha": 0.000055},
}
feat_params = {
    "test": {"featcls": "Polynomial"},
    "test2": {"featcls": "Fourier"},
    "test3": {"featcls": "Polynomial", "degree": 3},
    "testweak": {"featcls": "WeakPDELibrary"} # needs work
}
opt_params = {
    "test": {"optcls": "STLSQ"},
    "miosr": {"optcls": "MIOSR"},
    "enslsq": ParamDetails({"optcls": "ensemble", "opt": ps.STLSQ(), "bagging": True, "n_models":20}, [ps])
}

# Grid search parameters
metrics = {
    "test": ["coeff_f1", "coeff_mae"],
    "lorenzk": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae"],
}
other_params = {
    "test": {
        "sim_params": sim_params["test"],
        "diff_params": diff_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    },
    "lorenzk": {
        "sim_params": sim_params["test"],
        "diff_params": diff_params["kalman"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    },
}
grid_params = {
    "test": ["sim_params.t_end"],
    "lorenzk": ["sim_params.t_end", "sim_params.noise_stdev", "diff_params.alpha"],
}
grid_vals = {
    "test": [[5, 10, 15, 20]],
    "lorenzk": ParamDetails([[1, 3, 9, 27], [.1, .3, .8], np.logspace(-6,-1,7)], [np])
}
grid_decisions = {
    "test": ["plot"],
    "lorenzk": ["plot", "plot", "max"]
}
