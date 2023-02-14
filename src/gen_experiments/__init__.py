from mitosis import Parameter

from . import sho
from . import lorenz
from . import nonlinear_pendulum
from . import cubic_oscillator
from . import vanderpol
from . import hopf
from . import odes
from . import lorenz_missing

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
}


def lookup_params(ex_name: str, params: list):
    resolved_params = []
    for param in params:
        p_name, p_id = param.split("=")
        choices = globals()[p_name]
        resolved_params.append(Parameter(p_id, p_name, choices[p_id]))
    return resolved_params


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
