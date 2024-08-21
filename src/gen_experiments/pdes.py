import itertools
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

from . import config
from .data import gen_pde_data
from .plotting import compare_coefficient_plots, plot_pde_training_data
from .utils import (
    FullSINDyTrialData,
    SINDyTrialData,
    coeff_metrics,
    integration_metrics,
    make_model,
    unionize_coeff_matrices,
)

logger = getLogger(__name__)
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


def diffuse1D_dirichlet(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)


def diffuse1D_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)


def burgers1D_dirichlet(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((uxx - u * ux), nx)


def burgers1D_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((0.1 * uxx - u * ux), nx)


def ks_dirichlet(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = ps.differentiation.SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx - uxxxx - u * ux, nx)


def ks_periodic(t, u, dx, nx):
    u = np.reshape(u, nx)
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = ps.differentiation.SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = ps.differentiation.SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx - uxxxx - u * ux, nx)


def kdv(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    ux = ps.differentiation.SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxxx = ps.differentiation.SpectralDerivative(d=3, axis=0)._differentiate(u, dx)
    return np.reshape(6 * u * ux - uxxx, nx)


pde_setup = {
    "diffuse1D_dirichlet": {
        "rhsfunc": {"func": diffuse1D_dirichlet, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 10],
        "coeff_true": [{"u_11": 1}],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
    "diffuse1D_periodic": {
        "rhsfunc": {"func": diffuse1D_periodic, "dimension": 1},
        "input_features": ["u"],
        "coeff_true": [{"u_11": 1}],
        "spatial_grid": np.linspace(-8, 8, 256),
    },
    "burgers1D_dirichlet": {
        "rhsfunc": {"func": burgers1D_dirichlet, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 10],
        "coeff_true": [{"u_11": 1, "uu_1": 1}],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
    "burgers1D_periodic": {
        "rhsfunc": {"func": burgers1D_periodic, "dimension": 1},
        "input_features": ["u"],
        "coeff_true": [{"u_11": 0.1, "uu_1": -1}],
        "spatial_grid": np.linspace(-8, 8, 256),
    },
    "ks_dirichlet": {
        "rhsfunc": {"func": ks_dirichlet, "dimension": 1},
        "input_features": ["u"],
        "time_args": [0.1, 10],
        "coeff_true": [
            {"u_11": -1, "u_1111": -1, "uu_1": -1},
        ],
        "spatial_grid": np.arange(0, 10, 0.1),
    },
    "ks_periodic": {
        "rhsfunc": {"func": ks_periodic, "dimension": 1},
        "input_features": ["u"],
        "coeff_true": [
            {"u_11": -1, "u_1111": -1, "uu_1": -1},
        ],
        "spatial_grid": np.linspace(0, 100, 1024),
    },
}


def data_prep(
    seed,
    group,
    sim_params,
    grid_params=None,
    grid_vals=None,
):
    rhsfunc = pde_setup[group]["rhsfunc"]["func"]
    dimension = pde_setup[group]["rhsfunc"]["dimension"]
    spatial_grid = pde_setup[group]["spatial_grid"]
    initial_condition = sim_params["init_cond"]
    dt = sim_params["dt"]
    spatial_args = [
        (spatial_grid[-1] - spatial_grid[0]) / len(spatial_grid),
        len(spatial_grid),
    ]
    if grid_params is not None:
        full_data = {}
        for combo in itertools.product(*grid_vals):
            t_end = combo[0]
            rel_noise = combo[1]
            key = f"data_{combo}"
            full_data[key] = gen_pde_data(
                rhsfunc,
                initial_condition,
                spatial_args,
                dimension,
                seed,
                noise_abs=None,
                noise_rel=rel_noise,
                dt=dt,
                t_end=t_end,
            )
        return {"main": None, "data": full_data}
    else:
        rel_noise = sim_params["rel_noise"]
        t_end = sim_params["t_end"]
        data = gen_pde_data(
            rhsfunc,
            initial_condition,
            spatial_args,
            dimension,
            seed,
            noise_abs=None,
            noise_rel=rel_noise,
            dt=dt,
            t_end=t_end,
        )
        return {"main": None, "data": data}


def run(
    seed: float,
    data: dict,
    group: str,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, SINDyTrialData | FullSINDyTrialData]:
    input_features = pde_setup[group]["input_features"]
    coeff_true = pde_setup[group]["coeff_true"]
    rel_noise = sim_params["rel_noise"]
    t_end = sim_params["t_end"]
    search_key = f"data_{t_end, rel_noise}"
    if search_key in data:
        dt = data[search_key]["dt"]
        t_train = data[search_key]["t_train"]
        x_train = data[search_key]["x_train"]
        x_test = data[search_key]["x_test"]
        x_dot_test = data[search_key]["x_dot_test"]
        x_train_true = data[search_key]["x_train_true"]
        rel_noise = data[search_key]["rel_noise"]
    else:
        raise KeyError(f"{search_key} not found in data")

    model = make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, t=t_train)
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    sim_ind = -1
    trial_data: SINDyTrialData = {
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
        "rel_noise": rel_noise,
    }
    if display:
        plot_pde_panel(trial_data)

    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return (metrics, trial_data)
    return metrics


def plot_pde_panel(trial_data: FullSINDyTrialData):
    trial_data["model"].print()
    plot_pde_training_data(
        trial_data["x_train"],
        trial_data["x_true"],
        trial_data["smooth_train"],
        trial_data["rel_noise"],
    )
    compare_coefficient_plots(
        trial_data["coeff_fit"],
        trial_data["coeff_true"],
        input_features=trial_data["input_features"],
        feature_names=trial_data["feature_names"],
    )
    plt.show()
