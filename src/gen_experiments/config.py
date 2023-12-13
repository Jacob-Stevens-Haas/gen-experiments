import numpy as np
import pysindy as ps

import gen_experiments
from gen_experiments import utils
from gen_experiments.utils import (
    FullTrialData,
    NestedDict,
    ParamDetails,
    SeriesDef,
    SeriesList,
    _PlotPrefs,
    _signal_avg_power,
)

ND = lambda d: NestedDict(**d)

def _convert_abs_rel_noise(
    grid_vals: list, grid_params: list, recent_results: FullTrialData
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
    return x*x


def addn(x):
    return x+x


plot_prefs = {
    "test": ParamDetails(_PlotPrefs(True, False, ({"sim_params.t_end": 10},)), [utils, gen_experiments]),
    "test-absrel": ParamDetails(
        _PlotPrefs(True, _convert_abs_rel_noise, ({"sim_params.noise_abs": 1},)),
        [utils, gen_experiments],
    ),
    "test-absrel2": ParamDetails(
        _PlotPrefs(
            True,
            _convert_abs_rel_noise,
            (
                {"sim_params.noise_abs": 0.1},
                {"sim_params.noise_abs": 0.5},
                {"sim_params.noise_abs": 1},
                {"sim_params.noise_abs": 2},
                {"sim_params.noise_abs": 4},
                {"sim_params.noise_abs": 8},
            ),
        ),
        [utils, gen_experiments],
    ),
    "test-absrel3": ParamDetails(
        _PlotPrefs(
            True,
            _convert_abs_rel_noise,
            (
                {
                    "sim_params.noise_abs": 1,
                    "diff_params.smoother_kws.window_length": 15,
                },
                {"sim_params.noise_abs": 1, "diff_params.meas_var": 1},
                {"sim_params.noise_abs": 1, "diff_params.alpha": 1e-2},
            ),
        ),
        [utils, gen_experiments],
    ),
    "test-absrel4": ParamDetails(
        _PlotPrefs(
            True,
            _convert_abs_rel_noise,
            (
                {
                    "sim_params.noise_abs": 1,
                    "diff_params.smoother_kws.window_length": 15,
                },
                {"sim_params.noise_abs": 1, "diff_params.meas_var": 1},
                {"sim_params.noise_abs": 1, "diff_params.alpha": 1e0},
                {
                    "sim_params.noise_abs": 2,
                    "diff_params.smoother_kws.window_length": 15,
                },
                {"sim_params.noise_abs": 2, "diff_params.meas_var": 4},
                {"sim_params.noise_abs": 2, "diff_params.alpha": 1e-1},
            ),
        ),
        [utils, gen_experiments],
    ),
    "test-absrel5": ParamDetails(
        _PlotPrefs(
            True,
            _convert_abs_rel_noise,
            (
                {
                    "sim_params.noise_abs": 1,
                    "diff_params.diffcls": "SmoothedFiniteDifference",
                },
                {"sim_params.noise_abs": 1, "diff_params.kind": "kalman"},
                {"sim_params.noise_abs": 1, "diff_params.kind": "total_variation"},
                {
                    "sim_params.noise_abs": 2,
                    "diff_params.diffcls": "SmoothedFiniteDifference",
                },
                {"sim_params.noise_abs": 2, "diff_params.kind": "kalman"},
                {"sim_params.noise_abs": 2, "diff_params.kind": "total_variation"},
                {
                    "sim_params.noise_abs": 4,
                    "diff_params.diffcls": "SmoothedFiniteDifference",
                },
                {"sim_params.noise_abs": 4, "diff_params.kind": "kalman"},
                {"sim_params.noise_abs": 4, "diff_params.kind": "total_variation"},
            ),
            {(0, 2), (3, 2), (0, 3), (3, 3), (0, 4), (3, 4)}
        ),
        [utils, gen_experiments],
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
    "pde2": ParamDetails(
        ND({
            "featcls": "pde",
            "library_functions": [identity, quadratic],
            "function_names": [identity, addn],
            "derivative_order": 2,
            "spatial_grid": np.arange(0, 10, 0.1),
            "include_interaction": True
        }),[ps]
    ),
    "pde3": ParamDetails(
        ND({
            "featcls": "pde",
            "library_functions": [identity, quadratic],
            "function_names": [identity, addn],
            "derivative_order": 3,
            "spatial_grid": np.arange(0, 10, 0.1),
            "include_interaction": True,
            "is_uniform": True,
        }),[ps]
    ),
    "pde4": ParamDetails(
        ND({
            "featcls": "pde",
            "library_functions": [identity, quadratic],
            "function_names": [identity, addn],
            "derivative_order": 4,
            "spatial_grid": np.arange(0, 10, 0.1),
            "include_interaction": True,
            "is_uniform": True,
            "periodic": True,
            "include_bias": True,
        }),[ps]
    )
}
opt_params = {
    "test": ND({"optcls": "STLSQ"}),
    "miosr": ND({"optcls": "MIOSR"}),
    "enslsq": ParamDetails(
        ND({"optcls": "ensemble", "opt": ps.STLSQ(), "bagging": True, "n_models": 20}),
        [ps],
    ),
    "ensmio-ho-vdp-lv-duff": ParamDetails(
        ND(
            {
                "optcls": "ensemble",
                "opt": ps.MIOSR(target_sparsity=4),
                "bagging": True,
                "n_models": 20,
            }
        ),
        [ps],
    ),
    "ensmio-hopf": ParamDetails(
        ND(
            {
                "optcls": "ensemble",
                "opt": ps.MIOSR(target_sparsity=8),
                "bagging": True,
                "n_models": 20,
            }
        ),
        [ps],
    ),
    "ensmio-lorenz-ross": ParamDetails(
        ND(
            {
                "optcls": "ensemble",
                "opt": ps.MIOSR(target_sparsity=7),
                "bagging": True,
                "n_models": 20,
            }
        ),
        [ps],
    ),
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
    "test-kalman-heuristic2": ND(
        {
            "sim_params": sim_params["test"],
            "diff_params": diff_params["kalman-empty2"],
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
    "exp1": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["test"],
                "opt_params": opt_params["enslsq"].vals,
            }
        ),
        [ps],
    ),
    "cubic": ND(
        {
            "sim_params": sim_params["test"],
            "feat_params": feat_params["cubic"],
            "opt_params": opt_params["test"],
        }
    ),
    "exp2": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["enslsq"].vals,
            }
        ),
        [ps],
    ),
    "abs-exp3": ParamDetails(
        ND(
            {
                "sim_params": sim_params["med-noise-many"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["ensmio-lorenz-ross"].vals,
            }
        ),
        [ps],
    ),
    "rel-exp3-lorenz": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["ensmio-lorenz-ross"].vals,
            }
        ),
        [ps],
    ),
    "lor-ross-cubic": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["ensmio-lorenz-ross"].vals,
            }
        ),
        [ps],
    ),
    "lor-ross-cubic-fast": ParamDetails(
        ND(
            {
                "sim_params": sim_params["test"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["mio-lorenz-ross"],
            }
        ),
        [ps],
    ),
    "4nonzero-cubic": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["ensmio-ho-vdp-lv-duff"].vals,
            }
        ),
        [ps],
    ),
    "hopf-cubic": ParamDetails(
        ND(
            {
                "sim_params": sim_params["10x"],
                "feat_params": feat_params["cubic"],
                "opt_params": opt_params["ensmio-hopf"].vals,
            }
        ),
        [ps],
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
    "abs_noise": [[0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman": [[0.1, 0.5, 1, 2, 4, 8], [0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman2": [[0.1, 0.5, 1, 2, 4, 8], [0.01, 0.25, 1, 4, 16, 64]],
    "tv1": ParamDetails([np.logspace(-4, 0, 5)], [np]),
    "tv2": ParamDetails([np.logspace(-3, -1, 5)], [np]),
    "lorenzk": ParamDetails([[1, 9, 27], [0.1, 0.8], np.logspace(-6, -1, 4)], [np]),
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
    "auto-kalman": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.1, 0.5, 1, 2, 4, 8)],
    ),
    "auto-kalman2": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.01, 0.25, 1, 4, 16, 64)],
    ),
    "auto-kalman3": SeriesDef(
        "Kalman",
        diff_params["kalman"],
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
                diff_series["auto-kalman"],
                diff_series["tv2"],
                diff_series["sg2"],
            ],
        ),
        [np],
    ),
    "kalman-auto2": ParamDetails(
        SeriesList(
            "diff_params",
            "Differentiation Method",
            [
                diff_series["auto-kalman2"],
                diff_series["tv2"],
                diff_series["sg2"],
            ],
        ),
        [np],
    ),
    "kalman-auto3": ParamDetails(
        SeriesList(
            "diff_params",
            "Differentiation Method",
            [
                diff_series["auto-kalman3"],
                diff_series["tv2"],
                diff_series["sg2"],
            ],
        ),
        [np],
    ),
}


skinny_specs = {
    "exp3": ParamDetails(
        (("sim_params.noise_abs", "diff_params.meas_var"), ((identity,), (identity,))),
        [gen_experiments],
    ),
    "abs_noise-kalman": ParamDetails(
        (tuple(grid_params["abs_noise-kalman"]), ((identity,), (identity,))),
        [gen_experiments],
    ),
    "duration-noise-kalman": ParamDetails(
        (
            ("sim_params.t_end", "sim_params.noise_abs", "diff_params.meas_var"),
            (
                (1, 1),
                (
                    -1,
                    identity,
                ),
                (
                    -1,
                    identity,
                ),
            ),
        ),
        [gen_experiments],
    ),
    "duration-noise": ParamDetails(
        (
            ("sim_params.t_end", "sim_params.noise_abs"),
            (
                (1,),
                (-1,),
            ),
        ),
        [gen_experiments],
    ),
}
