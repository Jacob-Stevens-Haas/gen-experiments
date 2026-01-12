from functools import partial
from logging import getLogger
from typing import Callable, Optional, TypeVar, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from pysindy._core import _BaseSINDy

from . import config
from .plotting import (
    compare_coefficient_plots,
    plot_test_trajectories,
    plot_training_data,
)
from .typing import ProbData
from .utils import (
    FullSINDyTrialData,
    SINDyTrialData,
    coeff_metrics,
    integration_metrics,
    make_model,
    simulate_test_data,
    unionize_coeff_matrices,
)

name = "odes"
lookup_dict = vars(config)
metric_ordering = {
    "coeff_precision": "max",
    "coeff_f1": "max",
    "coeff_recall": "max",
    "coeff_mae": "min",
    "coeff_mse": "min",
    "mse_plot": "min",
    "mae_plot": "min",
}


T = TypeVar("T", bound=int)
DType = TypeVar("DType", bound=np.dtype)
MOD_LOG = getLogger(__name__)


def add_forcing(
    forcing_func: Callable[[float], np.ndarray[tuple[T], DType]],
    auto_func: Callable[
        [float, np.ndarray[tuple[T], DType]], np.ndarray[tuple[T], DType]
    ],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Add a time-dependent forcing term to a rhs func

    Args:
        forcing_func: The forcing function to add
        auto_func: An existing rhs func for solve_ivp

    Returns:
        A rhs function for integration
    """

    def sum_of_terms(
        t: float, state: np.ndarray[tuple[T], DType]
    ) -> np.ndarray[tuple[T], DType]:
        return np.array(forcing_func(t)) + np.array(auto_func(t, state))

    return sum_of_terms


def nonlinear_pendulum(
    t, x, m=1, L=1, g=9.81, forcing=0, return_all=True
):  # type:ignore
    """Simple pendulum equation of motion

    Arguments:
        t (float): ignored if system and forcing is autonomous
        x ([np.array, Sequence]): angular position and velocity
        m (float): mass of pendulum weight in kilograms
        L (float): length of pendulum in meters
        g (float): gravitational acceleration in :math:`m/s^2`.
        forcing ([float, Callable]): Constant forcing or forcing
            function.  If function, accepts arguments (t, x).
    """
    if not isinstance(forcing, Callable):
        const_force = forcing

        def forcing(t, x):
            return const_force

    moment_of_inertia = m * L**2
    return (x[1], (-m * g * np.sin(x[0]) + forcing(t, x)) / moment_of_inertia)


p_duff = [0.2, 0.05, 1]
p_lotka = [5, 1]
p_ross = [0.2, 0.2, 5.7]
p_hopf = [-0.05, 1, 1]

ode_setup = {
    "duff": {
        "rhsfunc": ps.utils.odes.duffing,
        "input_features": ["x", "x'"],
        "coeff_true": [
            {"x'": 1},
            {"x'": -p_duff[0], "x": -p_duff[1], "x^3": -p_duff[2]},
        ],
    },
    "lv": {
        "rhsfunc": partial(ps.utils.odes.lotka, p=p_lotka),
        "input_features": ["x", "y"],
        "coeff_true": [
            {"x": p_lotka[0], "x y": -p_lotka[1]},
            {"y": -2 * p_lotka[0], "x y": p_lotka[1]},
        ],
        "x0_center": 5 * np.ones(2),
        "nonnegative": True,
    },
    "ross": {
        "rhsfunc": ps.utils.odes.rossler,
        "input_features": ["x", "y", "z"],
        "coeff_true": [
            {"y": -1, "z": -1},
            {"x": 1, "y": p_ross[0]},
            {"1": p_ross[1], "z": -p_ross[2], "x z": 1},
        ],
    },
    "lorenz": {
        "rhsfunc": ps.utils.lorenz,
        "input_features": ["x", "y", "z"],
        "coeff_true": [
            {"x": -10, "y": 10},
            {"x": 28, "y": -1, "x z": -1},
            {"z": -8 / 3, "x y": 1},
        ],
        "x0_center": np.array([0, 0, 15]),
    },
    "lorenz_sin_forced": {
        "rhsfunc": add_forcing(lambda t: [50 * np.sin(t), 0, 0], ps.utils.lorenz),
        "input_features": ["x", "y", "z"],
        "coeff_true": [
            {"x": -10, "y": 10, "sin(t)": 50},
            {"x": 28, "y": -1, "x z": -1},
            {"z": -8 / 3, "x y": 1},
        ],
        "x0_center": np.array([0, 0, 15]),
    },
    "hopf": {
        "rhsfunc": ps.utils.hopf,
        "input_features": ["x", "y"],
        "coeff_true": [
            {"x": p_hopf[0], "y": -p_hopf[1], "x^3": -p_hopf[2], "x y^2": -p_hopf[2]},
            {"x": p_hopf[1], "y": p_hopf[0], "x^2 y": -p_hopf[2], "y^3": -p_hopf[2]},
        ],
    },
    "sho": {
        "rhsfunc": ps.utils.linear_damped_SHO,
        "input_features": ["x", "y"],
        "coeff_true": [
            {"x": -0.1, "y": 2},
            {"x": -2, "y": -0.1},
        ],
    },
    "cubic_ho": {
        "rhsfunc": ps.utils.cubic_damped_SHO,
        "input_features": ["x", "y"],
        "coeff_true": [
            {"x^3": -0.1, "y^3": 2},
            {"x^3": -2, "y^3": -0.1},
        ],
    },
    "vdp": {
        "rhsfunc": ps.utils.van_der_pol,
        "input_features": ["x", "x'"],
        "coeff_true": [
            {"x'": 1},
            {"x": -1, "x'": 0.5, "x^2 x'": -0.5},
        ],
    },
    "kinematics": {
        "rhsfunc": lambda t, x: [x[1], -1],
        "input_features": ["x", "x'"],
        "coeff_true": [
            {"x'": 1},
            {"1": -1},
        ],
    },
    "pendulum": {
        "rhsfunc": nonlinear_pendulum,
        "input_features": ["x", "x'"],
        "coeff_true": [
            {"x'": 1},
            {"sin(1 x)": -9.81},
        ],
    },
    "lorenz_xy": {
        "rhsfunc": ps.utils.lorenz,
        "input_features": ["x", "y", "z"],
        "coeff_true": [
            {"x": -10, "y": 10},
            {"x": 28, "y": -1},
        ],
    },
}


def run(
    data: ProbData,
    diff_params: Optional[dict] = None,
    feat_params: Optional[dict] = None,
    opt_params: Optional[dict] = None,
    model: Optional[ps._core._BaseSINDy] = None,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, SINDyTrialData | FullSINDyTrialData]:
    input_features = data.input_features
    dt = data.dt
    x_train = data.x_train
    t_train = data.t_train
    x_train_true = data.x_train_true
    x_test = data.x_test
    x_dot_test = data.x_dot_test
    coeff_true = data.coeff_true

    if isinstance(feat_params, dict) and feat_params["featcls"] == "weak":
        feat_params.pop("featcls")
        feat_params = ps.WeakPDELibrary(**feat_params, spatiotemporal_grid=data.t_train)

    if model is None and all(
        (diff_params is not None, feat_params is not None, opt_params is not None)
    ):
        warn(
            "Passing the built model directly is now accepted.  This is the recommended"
            "way, and passing (nested) dictionaries to build classes is likely to be"
            "deprecated.",
            PendingDeprecationWarning,
        )
        model = make_model(input_features, dt, diff_params, feat_params, opt_params)
    elif (
        model is not None
        and any(
            (diff_params is not None, feat_params is not None, opt_params is not None)
        )
        or model is None
        and any((diff_params is None, feat_params is None, opt_params is None))
    ):
        raise ValueError(
            "Either model must be None and the builder dictionaries all defined,"
            "or vice versa."
        )
    model = cast(_BaseSINDy, model)
    model.feature_names = data.input_features
    model.fit(x_train, t=len(x_train) * [t_train])
    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(
        model, (data.input_features, coeff_true)
    )
    if isinstance(model.feature_library, ps.WeakPDELibrary):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.model.steps[0] = ("features", inner_lib)
        model.feature_library = inner_lib  # type: ignore  # TODO: Fix in pysindy
    sim_ind = -1
    if isinstance(model, ps.SINDy) and hasattr(
        model.differentiation_method, "smoothed_x_"
    ):
        smooth_x = model.differentiation_method.smoothed_x_
    else:  # using WeakPDELibrary
        smooth_x = x_train[0]
    trial_data: SINDyTrialData = {
        "dt": dt,
        "coeff_true": coeff_true,
        "coeff_fit": coefficients,
        "feature_names": feature_names,
        "input_features": input_features,
        "t_train": t_train,
        "x_true": x_train_true[sim_ind],
        "x_train": x_train[sim_ind],
        "smooth_train": smooth_x,
        "x_test": x_test[sim_ind],
        "x_dot_test": x_dot_test[sim_ind],
        "model": model,
    }
    if display:
        MOD_LOG.info(f"Simulating a model: {model}")
        trial_data: FullSINDyTrialData = trial_data | simulate_test_data(
            trial_data["model"], trial_data["dt"], trial_data["x_test"]
        )
        plot_ode_panel(trial_data)
    MOD_LOG.info(f"Evaluating a model: {model}")
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}
    return metrics


def ablate_feat(
    data: ProbData,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, SINDyTrialData | FullSINDyTrialData]:
    """Like run(), but hide one input feature from model

    Temporary and highly WET.
    """
    input_features = data.input_features[:-1]
    dt = data.dt
    x_train = [x[..., :-1] for x in data.x_train]
    t_train = data.t_train
    x_train_true = [x[..., :-1] for x in data.x_train_true]
    x_test = [x[..., :-1] for x in data.x_test]
    x_dot_test = [x[..., :-1] for x in data.x_dot_test]

    if feat_params["featcls"] == "weak":
        feat_params.pop("featcls")
        feat_params = ps.WeakPDELibrary(**feat_params, spatiotemporal_grid=data.t_train)

    model = make_model(input_features, dt, diff_params, feat_params, opt_params)
    model.fit(x_train, t=dt)
    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(
        model, (data.input_features[:-1], data.coeff_true[:-1])
    )
    if isinstance(model.feature_library, ps.WeakPDELibrary):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.model.steps[0] = ("features", inner_lib)
        model.feature_library = inner_lib
    sim_ind = -1
    if hasattr(model.differentiation_method, "smoothed_x_"):
        smooth_x = model.differentiation_method.smoothed_x_
    else:  # using WeakPDELibrary
        smooth_x = x_train[0]
    trial_data: SINDyTrialData = {
        "dt": dt,
        "coeff_true": coeff_true,
        "coeff_fit": coefficients,
        "feature_names": feature_names,
        "input_features": input_features,
        "t_train": t_train,
        "x_true": x_train_true[sim_ind],
        "x_train": x_train[sim_ind],
        "smooth_train": smooth_x,
        "x_test": x_test[sim_ind],
        "x_dot_test": x_dot_test[sim_ind],
        "model": model,
    }
    if display:
        MOD_LOG.info(f"Simulating a model: {model}")
        trial_data: FullSINDyTrialData = trial_data | simulate_test_data(
            trial_data["model"], trial_data["dt"], trial_data["x_test"]
        )
        plot_ode_panel(trial_data)
    MOD_LOG.info(f"Evaluating a model: {model}")
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}
    return metrics


def plot_ode_panel(trial_data: FullSINDyTrialData):
    trial_data["model"].print()
    plot_training_data(
        trial_data["x_train"], trial_data["x_true"], trial_data["smooth_train"]
    )
    compare_coefficient_plots(
        trial_data["coeff_fit"],
        trial_data["coeff_true"],
        input_features=[_texify(feat) for feat in trial_data["input_features"]],
        feature_names=[_texify(feat) for feat in trial_data["feature_names"]],
    )
    plot_test_trajectories(
        trial_data["x_test"],
        trial_data["x_sim"],
        trial_data["t_test"],
        trial_data["t_sim"],
    )
    plt.show()


def _texify(input: str) -> str:
    if input[0] != "$":
        input = "$" + input
    if input[-1] != "$":
        input = input + "$"
    return input
