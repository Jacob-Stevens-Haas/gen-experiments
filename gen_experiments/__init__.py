from mitosis import Parameter

from . import sho
from . import lorenz
from . import nonlinear_pendulum
from . import cubic_oscillator
from . import vanderpol
from . import hopf
from . import odes

experiments = {
    "sho": (sho, None),
    "lorenz": (lorenz, None),
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


sim_params = {"test": {"n_trajectories": 2}}
diff_params = {
    "test": {"kind": "FiniteDifference"},
    "test2": {"kind": "SmoothedFiniteDifference"},
}
feat_params = {
    "test": {"kind": "Polynomial"},
    "test2": {"kind": "Fourier"},
    "test3": {"kind": "Polynomial", "degree": 3},
}
opt_params = {"test": {"kind": "STLSQ"}}
