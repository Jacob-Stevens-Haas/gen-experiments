import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import SpectralDerivative

from . import config

from .utils import (
    TrialData,
    FullTrialData,
    gen_pde_data,
    compare_coefficient_plots,
    plot_pde_training_data,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model,
    plot_test_trajectories,
    simulate_test_data
)

name = "pdes"
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


def diffuse1D(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)


def burgers1D(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((uxx - u * ux), nx)


def ks(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx - uxxxx - u * ux, nx)


def kdv(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxxx = SpectralDerivative(d=3, axis=0)._differentiate(u, dx)
    return np.reshape(6 * u * ux - uxxx, nx)


pde_setup = {
    "diffuse1D": {
        "rhsfunc": {"func": diffuse1D, "dimension": 1},
        "input_features": ["u"],
        "initial_condition": 10 * np.exp(-((np.arange(0, 10, 0.1) - 5) ** 2) / 2),
        "spatial_args": [0.1, 100],
        "time_args": [0.1, 10],
        "coeff_true": [{"u_11": 1}],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
    "burgers1D": {
        "rhsfunc": {"func": burgers1D, "dimension": 1},
        "input_features": ["u"],
        "initial_condition": 10 * np.exp(-((np.arange(0, 10, 0.1) - 5) ** 2) / 2),
        "spatial_args": [0.1, 100],
        "time_args": [0.1, 10],
        "coeff_true": [{"u_11": 1, "uu_1": 1}],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
    "ks": {
        "rhsfunc": {"func": ks, "dimension": 1},
        "input_features": ["u"],
        "initial_condition": np.cos(np.arange(0, 10, 0.1)) * (
            1 + np.sin(np.arange(0, 10, 0.1))
        ),
        "spatial_args": [0.1, 100],
        "time_args": [0.1, 10],
        "coeff_true": [
            {"u_11": -1, "u_1111": -1, "uu_1": -1},
        ],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
}


def run(
    seed: float,
    /,
    group: str,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, TrialData | FullTrialData]:
    rhsfunc = pde_setup[group]["rhsfunc"]["func"]
    input_features = pde_setup[group]["input_features"]
    initial_condition = pde_setup[group]["initial_condition"]
    spatial_args = pde_setup[group]["spatial_args"]
    time_args = pde_setup[group]["time_args"]
    dimension = pde_setup[group]["rhsfunc"]["dimension"]
    coeff_true = pde_setup[group]["coeff_true"]
    try:
        time_args = pde_setup[group]["time_args"]
    except KeyError:
        time_args = [0.01, 10]
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_pde_data(
        rhsfunc,
        initial_condition,
        spatial_args,
        dimension,
        seed,
        noise_abs=0,
        dt=time_args[0],
        t_end=time_args[1],
    )
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, t=t_train)
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    sim_ind = -1
    trial_data: TrialData = {
                "dt": dt,
                "coeff_true": coeff_true,
                "coeff_fit": coefficients,
                "feature_names": feature_names,
                "input_features": input_features,
                "t_train": t_train,
                "x_true": x_train_true,
                "x_train": x_train[sim_ind],
                "smooth_train": model.differentiation_method.smoothed_x_,
                "x_test": x_test[sim_ind],
                "x_dot_test": x_dot_test[sim_ind],
                "model": model,
            }
    if display:
        trial_data: FullTrialData = trial_data | simulate_test_data(
            trial_data["model"], trial_data["dt"], trial_data["x_test"]
        )
        trial_data["model"].print()
        plot_pde_training_data(trial_data["x_train"], trial_data["x_true"], trial_data["smooth_train"])
        compare_coefficient_plots(
        trial_data["coeff_fit"],
        trial_data["coeff_true"],
        input_features=trial_data["input_features"],
        feature_names=trial_data["feature_names"],
        )

    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return (
            metrics, trial_data
        )
    return metrics