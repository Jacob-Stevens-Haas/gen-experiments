from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import pysindy as ps

from gen_experiments.data import _signal_avg_power
from gen_experiments.gridsearch.typing import (
    GridLocator,
    SeriesDef,
    SeriesList,
    SkinnySpecs,
)
from gen_experiments.plotting import _PlotPrefs
from gen_experiments.typing import NestedDict
from gen_experiments.utils import FullSINDyTrialData

T = TypeVar("T")
U = TypeVar("U")


def ND(d: dict[T, U]) -> NestedDict[T, U]:
    return NestedDict(**d)


def _convert_abs_rel_noise(
    grid_vals: list, grid_params: list, recent_results: FullSINDyTrialData
):
    """Convert abs_noise grid_vals to rel_noise"""
    signal = np.stack(recent_results["x_true"], axis=-1)
    signal_power = _signal_avg_power(signal)
    ind = grid_params.index("sim_params.noise_abs")
    grid_vals[ind] = grid_vals[ind] / signal_power
    grid_params[ind] = "sim_params.noise_rel"
    return grid_vals, grid_params


# To allow pickling
def identity(x):
    return x


def quadratic(x):
    return x * x


def addn(x):
    return x + x


plot_prefs = {
    "test": _PlotPrefs(),
    "test-absrel": _PlotPrefs(
        True, _convert_abs_rel_noise, GridLocator(..., {("sim_params.noise_abs", (1,))})
    ),
    "test-absrel2": _PlotPrefs(
        True,
        _convert_abs_rel_noise,
        GridLocator(
            ...,
            (..., ...),
            (
                {"sim_params.noise_abs": 0.1},
                {"sim_params.noise_abs": 0.5},
                {"sim_params.noise_abs": 1},
                {"sim_params.noise_abs": 2},
                {"sim_params.noise_abs": 4},
                {"sim_params.noise_abs": 8},
            ),
        ),
    ),
    "absrel-newloc": _PlotPrefs(
        True,
        _convert_abs_rel_noise,
        GridLocator(
            ["coeff_mse", "coeff_f1"],
            (..., (2, 3, 4)),
            (
                {"diff_params.kind": "kalman", "diff_params.alpha": None},
                {
                    "diff_params.kind": "kalman",
                    "diff_params.alpha": lambda a: isinstance(a, int),
                },
                {"diff_params.kind": "trend_filtered"},
                {"diff_params.diffcls": "SmoothedFiniteDifference"},
            ),
        ),
    ),
}
sim_params = {
    "test": ND({"n_trajectories": 2}),
    "test-r1": ND({"n_trajectories": 2, "noise_rel": 0.01}),
    "test-r2": ND({"n_trajectories": 2, "noise_rel": 0.1}),
    "test-r3": ND({"n_trajectories": 2, "noise_rel": 0.3}),
    "10x": ND({"n_trajectories": 10}),
    "10x-r1": ND({"n_trajectories": 10, "noise_rel": 0.01}),
    "10x-r2": ND({"n_trajectories": 10, "noise_rel": 0.05}),
    "test2": ND({"n_trajectories": 2, "noise_abs": 0.4}),
    "med-noise": ND({"n_trajectories": 2, "noise_abs": 0.8}),
    "med-noise-many": ND({"n_trajectories": 10, "noise_abs": 0.8}),
    "hi-noise": ND({"n_trajectories": 2, "noise_abs": 2}),
    "pde-ic1": ND({"init_cond": np.exp(-((np.arange(0, 10, 0.1) - 5) ** 2) / 2)}),
    "pde-ic2": ND({
        "init_cond": (np.cos(np.arange(0, 10, 0.1))) * (
            1 + np.sin(np.arange(0, 10, 0.1) - 0.5)
        )
    }),
}
diff_params = {
    "test": ND({"diffcls": "FiniteDifference"}),
    "autoks": ND({"diffcls": "sindy", "kind": "kalman", "alpha": "gcv"}),
    "test_axis": ND({"diffcls": "FiniteDifference", "axis": -2}),
    "test2": ND({"diffcls": "SmoothedFiniteDifference"}),
    "tv": ND({"diffcls": "sindy", "kind": "trend_filtered", "order": 0, "alpha": 1}),
    "savgol": ND({"diffcls": "sindy", "kind": "savitzky_golay"}),
    "sfd-nox": ND({"diffcls": "SmoothedFiniteDifference", "save_smooth": False}),
    "sfd-ps": ND({"diffcls": "SmoothedFiniteDifference"}),
    "kalman": ND({"diffcls": "sindy", "kind": "kalman", "alpha": 0.000055}),
    "kalman-empty2": ND({"diffcls": "sindy", "kind": "kalman", "alpha": None}),
    "kalman-auto": ND(
        {"diffcls": "sindy", "kind": "kalman", "alpha": None, "meas_var": 0.8}
    ),
}
feat_params = {
    "test": ND({"featcls": "Polynomial"}),
    "test2": ND({"featcls": "Fourier"}),
    "cubic": ND({"featcls": "Polynomial", "degree": 3}),
    "testweak": ND({"featcls": "WeakPDELibrary"}),  # needs work
    "pde2": ND({
        "featcls": "pde",
        "library_functions": [identity, quadratic],
        "function_names": [identity, addn],
        "derivative_order": 2,
        "spatial_grid": np.arange(0, 10, 0.1),
        "include_interaction": True,
    }),
    "pde3": ND({
        "featcls": "pde",
        "library_functions": [identity, quadratic],
        "function_names": [identity, addn],
        "derivative_order": 3,
        "spatial_grid": np.arange(0, 10, 0.1),
        "include_interaction": True,
        "is_uniform": True,
    }),
    "pde4": ND({
        "featcls": "pde",
        "library_functions": [identity, quadratic],
        "function_names": [identity, addn],
        "derivative_order": 4,
        "spatial_grid": np.arange(0, 10, 0.1),
        "include_interaction": True,
        "is_uniform": True,
        "periodic": True,
        "include_bias": True,
    }),
}
opt_params = {
    "test": ND({"optcls": "STLSQ"}),
    "miosr": ND({"optcls": "MIOSR"}),
    "enslsq": ND(
        {"optcls": "ensemble", "opt": ps.STLSQ(), "bagging": True, "n_models": 20}
    ),
    "ensmio-ho-vdp-lv-duff": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=4),
        "bagging": True,
        "n_models": 20,
    }),
    "ensmio-hopf": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=8),
        "bagging": True,
        "n_models": 20,
    }),
    "ensmio-lorenz-ross": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=7),
        "bagging": True,
        "n_models": 20,
    }),
    "mio-lorenz-ross": ND({"optcls": "MIOSR", "target_sparsity": 7}),
}

# Grid search parameters
metrics = {
    "test": ["coeff_f1", "coeff_mae"],
    "all-coeffs": ["coeff_f1", "coeff_mae", "coeff_mse"],
    "all": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae", "coeff_mse"],
    "lorenzk": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae"],
    "1": ["coeff_f1", "coeff_precision", "coeff_mse", "coeff_mae"],
}
other_params = {
    "test": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "tv1": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["tv"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test2": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test-kalman-heuristic2": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["kalman-empty2"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "lorenzk": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["kalman"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "exp1": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["enslsq"],
    }),
    "cubic": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["test"],
    }),
    "exp2": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["enslsq"],
    }),
    "abs-exp3": ND({
        "sim_params": sim_params["med-noise-many"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "rel-exp3-lorenz": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "lor-ross-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "lor-ross-cubic-fast": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["mio-lorenz-ross"],
    }),
    "4nonzero-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-ho-vdp-lv-duff"],
    }),
    "hopf-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-hopf"],
    }),
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
grid_vals: dict[str, list[Iterable]] = {
    "test": [[5, 10, 15, 20]],
    "abs_noise": [[0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman": [[0.1, 0.5, 1, 2, 4, 8], [0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman2": [[0.1, 0.5, 1, 2, 4, 8], [0.01, 0.25, 1, 4, 16, 64]],
    "tv1": [np.logspace(-4, 0, 5)],
    "tv2": [np.logspace(-3, -1, 5)],
    "lorenzk": [[1, 9, 27], [0.1, 0.8], np.logspace(-6, -1, 4)],
    "lorenz1": [[1, 3, 9, 27], [0.01, 0.1, 1]],
    "duration-absnoise": [[0.5, 1, 2, 4, 8, 16], [0.1, 0.5, 1, 2, 4, 8]],
    "rel_noise": [[0.25, 1, 4, 16], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]],
}
grid_decisions = {
    "test": ["plot"],
    "plot1": ["plot", "max"],
    "lorenzk": ["plot", "plot", "max"],
    "plot2": ["plot", "plot"],
}
diff_series: dict[str, SeriesDef] = {
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
    "auto-kalman": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.1, 0.5, 1, 2, 4, 8)],
    ),
    "auto-kalman2": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.01, 0.25, 1, 4, 16, 64)],
    ),
    "auto-kalman3": SeriesDef(
        "Auto Kalman",
        diff_params["kalman-auto"],
        ["diff_params.alpha"],
        [(None,)],
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
series_params: dict[str, SeriesList] = {
    "test": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["kalman1"],
            diff_series["tv1"],
            diff_series["sg1"],
        ],
    ),
    "lorenz1": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto2": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto3": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman3"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "multikalman": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman3"],
            diff_series["kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
}


skinny_specs: dict[str, SkinnySpecs] = {
    "exp3": (
        ("sim_params.noise_abs", "diff_params.meas_var"),
        ((identity,), (identity,)),
    ),
    "abs_noise-kalman": (
        tuple(grid_params["abs_noise-kalman"]),
        ((identity,), (identity,)),
    ),
    "duration-noise-kalman": (
        ("sim_params.t_end", "sim_params.noise_abs", "diff_params.meas_var"),
        ((1, 1), (-1, identity), (-1, identity)),
    ),
    "duration-noise": (("sim_params.t_end", "sim_params.noise_abs"), ((1,), (-1,))),
}
