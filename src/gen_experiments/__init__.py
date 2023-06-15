import importlib

import numpy as np
import pysindy as ps

from mitosis import Parameter

from . import nonlinear_pendulum
from . import odes
from . import lorenz_missing
from . import gridsearch
from .utils import (
    NestedDict, ParamDetails, SeriesDef, SeriesList, _PlotPrefs, _max_amplitude
)
from . import utils

experiments = {
    "sho": (odes, "sho"),
    "lorenz": (odes, "lorenz"),
    "lorenz_2d": (lorenz_missing, None),
    "pendulum": (nonlinear_pendulum, None),
    "cubic_ho": (odes, "cubic_ho"),
    "vdp": (odes, "vdp"),
    "hopf": (odes, "hopf"),
    "duff": (odes, "duff"),
    "lv": (odes, "lv"),
    "ross": (odes, "ross"),
    "gridsearch": (gridsearch, None),
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


def _convert_abs_rel_noise(grid_vals, grid_params, recent_results):
    """Convert abs_noise grid_vals to rel_noise"""
    signal = recent_results["x_train_true"]
    signal_amplitude = _max_amplitude(signal)
    ind = grid_params.index("sim_params.noise_abs")
    grid_vals[ind] = grid_vals[ind]/signal_amplitude
    grid_params[ind] = "sim_params.noise_rel"
    return grid_vals, grid_params


ND = lambda d: NestedDict(**d)
plot_prefs = {
    "test": _PlotPrefs(True, False, ({"sim_params.t_end": 20},)),
    "test-absrel": ParamDetails(
        _PlotPrefs(True, _convert_abs_rel_noise, ({"sim_params.noise_abs": 1}, )),
        [utils, importlib.import_module(__name__)]
    ),
    "test-absrel2": ParamDetails(
        _PlotPrefs(True, _convert_abs_rel_noise, (
            {"sim_params.noise_abs": .1},
            {"sim_params.noise_abs": .5},
            {"sim_params.noise_abs": 1},
            {"sim_params.noise_abs": 2},
            {"sim_params.noise_abs": 4},
            {"sim_params.noise_abs": 8},
        )),
        [utils, importlib.import_module(__name__)]
    ),
}
sim_params = {
    "test": ND({"n_trajectories": 2}),
    "test-r1": ND({"n_trajectories": 2, "noise_rel": .01}),
    "test-r2": ND({"n_trajectories": 2, "noise_rel": .1}),
    "test-r3": ND({"n_trajectories": 2, "noise_rel": .3}),
    "10x": ND({"n_trajectories": 10}),
    "10x-r1": ND({"n_trajectories": 10, "noise_rel": .01}),
    "10x-r2": ND({"n_trajectories": 10, "noise_rel": .05}),
    "test2": ND({"n_trajectories": 2, "noise_abs": 0.4}),
    "med-noise": ND({"n_trajectories": 2, "noise_abs": 0.8}),
    "med-noise-many": ND({"n_trajectories": 10, "noise_abs": 0.8}),
    "hi-noise": ND({"n_trajectories": 2, "noise_abs": 2}),
}
diff_params = {
    "test": ND({"diffcls": "FiniteDifference"}),
    "test2": ND({"diffcls": "SmoothedFiniteDifference"}),
    "tv": ND({"diffcls": "sindy", "kind": "trend_filtered", "order": 0, "alpha": 1}),
    "savgol": ND({"diffcls": "sindy", "kind": "savitzky_golay"}),
    "sfd-nox": ND({"diffcls": "SmoothedFiniteDifference", "save_smooth": False}),
    "sfd-ps": ND({"diffcls": "SmoothedFiniteDifference"}),
    "kalman": ND({"diffcls": "sindy", "kind": "kalman", "alpha": 0.000055}),
    "kalman-empty": ND({"diffcls": "sindy", "kind": "kalman"}),
    "kalman-auto": ND({"diffcls": "sindy", "kind": "kalman", "alpha": None, "meas_var": .8}),
}
feat_params = {
    "test": ND({"featcls": "Polynomial"}),
    "test2": ND({"featcls": "Fourier"}),
    "cubic": ND({"featcls": "Polynomial", "degree": 3}),
    "testweak": ND({"featcls": "WeakPDELibrary"}),  # needs work
}
opt_params = {
    "test": ND({"optcls": "STLSQ"}),
    "miosr": ND({"optcls": "MIOSR"}),
    "enslsq": ParamDetails(
        ND({"optcls": "ensemble", "opt": ps.STLSQ(), "bagging": True, "n_models": 20}),
        [ps],
    ),
    "ensmio-lorenz": ParamDetails(
        ND({"optcls": "ensemble", "opt": ps.MIOSR(target_sparsity=7), "bagging": True, "n_models": 20}),
        [ps],
    ),
}

# Grid search parameters
metrics = {
    "test": ["coeff_f1", "coeff_mae"],
    "lorenzk": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae"],
    "1": ["coeff_f1", "coeff_precision", "coeff_mse", "coeff_mae"],
}
other_params = {
    "test": ND(
        {
            "sim_params": sim_params["test"],
            "diff_params": diff_params["test"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["test"],
        }
    ),
    "tv1": ND(
        {
            "sim_params": sim_params["test"],
            "diff_params": diff_params["tv"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["test"],
        }
    ),
    "test2": ND(
        {
            "sim_params": sim_params["test"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["test"],
        }
    ),
    "test-kalman-heuristic": ND(
        {
            "sim_params": sim_params["test"],
            "diff_params": diff_params["kalman-empty"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["test"],
        }
    ),
    "lorenzk": ND(
        {
            "sim_params": sim_params["test"],
            "diff_params": diff_params["kalman"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["test"],
        }
    ),
    "exp1": ParamDetails(ND(
        {
            "sim_params": sim_params["10x"],
            "feat_params": feat_params["test"],
            "opt_params": opt_params["enslsq"].vals,
        }), [ps]
    ),
    "cubic": ND(
        {
            "sim_params": sim_params["test"],
            "feat_params": feat_params["cubic"],
            "opt_params": opt_params["test"],
        }
    ),
    "exp2": ParamDetails(ND(
        {
            "sim_params": sim_params["10x"],
            "feat_params": feat_params["cubic"],
            "opt_params": opt_params["enslsq"].vals,
        }), [ps]
    ),
    "abs-exp3": ParamDetails(ND(
        {
            "sim_params": sim_params["med-noise-many"],
            "feat_params": feat_params["cubic"],
            "opt_params": opt_params["ensmio-lorenz"].vals,
        }), [ps]
    ),
    "rel-exp3-lorenz": ParamDetails(ND(
        {
            "sim_params": sim_params["10x"],
            "feat_params": feat_params["cubic"],
            "opt_params": opt_params["ensmio-lorenz"].vals,
        }), [ps]
    ),
}
grid_params = {
    "test": ["sim_params.t_end"],
    "abs_noise": ["sim_params.noise_abs"],
    "abs_noise-kalman": ["sim_params.noise_abs", "diff_params.meas_var"],
    "tv1": ["diff_params.alpha"],
    "lorenzk": ["sim_params.t_end", "sim_params.noise_abs", "diff_params.alpha"],
    "duration-absnoise": ["sim_params.t_end", "sim_params.noise_abs"],
    "rel_noise": ["sim_params.t_end", "sim_params.noise_rel"],
}
grid_vals = {
    "test": [[5, 10, 15, 20]],
    "abs_noise": [[0.1, .5, 1, 2, 4, 8]],
    "abs_noise-kalman": [[0.1, .5, 1, 2, 4, 8], [0.1, .5, 1, 2, 4, 8]],
    "tv1": ParamDetails([np.logspace(-4, 0, 5)], [np]),
    "tv2": ParamDetails([np.logspace(-3, -1, 5)], [np]),
    "lorenzk": ParamDetails([[1, 9, 27], [0.1, 0.8], np.logspace(-6, -1, 4)], [np]),
    "lorenz1": [[1, 3, 9, 27], [0.01, 0.1, 1]],
    "lorenz2": [[.25, 1, 4, 16], [0.1, .5, 1, 2, 4, 8]],
    "rel_noise": [[.25, 1, 4, 16], [0.05, .1, .15, .2, .25, .3]],
}
grid_decisions = {
    "test": ["plot"],
    "plot1": ["plot", "max"],
    "lorenzk": ["plot", "plot", "max"],
    "plot2": ["plot", "plot"],
}
diff_series = {
    "kalman1": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [np.logspace(-6, 0, 3)],
    ),
    "kalman2": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [np.logspace(-4, 0, 5)],
    ),
    "lorenz-auto": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.1, .5, 1, 2, 4, 8)],
    ),
    "tv1": SeriesDef(
        "Total Variation",
        diff_params["tv"],
        ["diff_params.alpha"],
        [np.logspace(-6, 0, 3)],
    ),
    "tv2": SeriesDef(
        "Total Variation",
        diff_params["tv"],
        ["diff_params.alpha"],
        [np.logspace(-4, 0, 5)],
    ),
    "sg1": SeriesDef(
        "Savitsky-Golay",
        diff_params["sfd-ps"],
        ["diff_params.smoother_kws.window_length"],
        [[5, 7, 15]],
    ),
    "sg2": SeriesDef(
        "Savitsky-Golay",
        diff_params["sfd-ps"],
        ["diff_params.smoother_kws.window_length"],
        [[5, 8, 12, 15]],
    ),
}
series_params = {
    "test": ParamDetails(
        SeriesList(
            "diff_params",
            "Differentiation Method",
            [
                diff_series["kalman1"],
                diff_series["tv1"],
                diff_series["sg1"],
            ],
        ),
        [np],
    ),
    "lorenz1": ParamDetails(
        SeriesList(
            "diff_params",
            "Differentiation Method",
            [
                diff_series["kalman2"],
                diff_series["tv2"],
                diff_series["sg2"],
            ],
        ),
        [np],
    ),
    "kalman-auto": ParamDetails(
        SeriesList(
            "diff_params",
            "Differentiation Method",
            [
                diff_series["lorenz-auto"],
                diff_series["tv2"],
                diff_series["sg2"],
            ],
        ),
        [np],
    ),
}


# To allow pickling
def identity(x): return x


skinny_specs = {
    "exp3": ParamDetails(
        (
            ("sim_params.noise_abs", "diff_params.meas_var"),
            ((identity,), (identity,))
        ),
        [importlib.import_module(__name__)]
    ),
    "abs_noise-kalman": ParamDetails(
        (tuple(grid_params["abs_noise-kalman"]), ((identity,), (identity,))),
        [importlib.import_module(__name__)]
    )
}
