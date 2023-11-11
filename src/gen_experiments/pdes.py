import numpy as np
from pysindy.differentiation import SpectralDerivative
from .utils import (
    gen_pde_data,
    compare_coefficient_plots,
    plot_pde_training_data,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model,
)

name = "pdes"

def diffuse1D(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)

pde_setup = {
    "diffuse1D": {
        "rhsfunc": {
            "func": diffuse1D,
            "dimension": 1
        },
        "input_features": ["u"],
        "initial_condition": 10*np.exp(-(np.arange(0, 10, 0.1)-5)**2/2),
        "spatial_args": [0.1, 100],
        "time_args": [0.1, 10],
        "coeff_true": [
            {"u_11": 1}
        ],
        "spatial_grid": np.arange(0, 10, 0.1)
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
) -> dict:
    rhsfunc = pde_setup[group]["rhsfunc"]["func"]
    input_features = pde_setup[group]["input_features"]
    initial_condition = pde_setup[group]["initial_condition"]
    spatial_args = pde_setup[group]["spatial_args"]
    time_args = pde_setup[group]["time_args"]
    dimension = pde_setup[group]["rhsfunc"]["dimension"]
    coeff_true = pde_setup[group]["coeff_true"]
    spatial_grid = pde_setup[group]["spatial_grid"]
    try:
        time_args = pde_setup[group]["time_args"]
    except KeyError:
        time_args = [0.01, 10]
    nonnegative = False
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_pde_data(
        rhsfunc,
        initial_condition,
        spatial_args,
        dimension,
        seed,
        noise_abs=0,
        dt=time_args[0],
        t_end=time_args[1]
    )
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, t=t_train)
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    # make the plots
    if display:
        model.print()
        compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )
        smoothed_last_train = model.differentiation_method.smoothed_x_
        plot_pde_training_data(x_train[-1], x_train_true, smoothed_last_train)

    # calculate metrics
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return (
            metrics, {
                "t_train": t_train,
                "x_train": x_train,
                "x_test": x_test,
                "x_dot_test": x_dot_test,
                "x_train_true": x_train_true,
                "model": model,
            }
        )
    return metrics
