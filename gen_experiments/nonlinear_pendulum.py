from typing import Callable
import numpy as np

from .utils import (
    gen_data,
    compare_coefficient_plots,
    plot_training_data,
    plot_test_trajectories,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model
)

name = "pendulum"


def nonlinear_pendulum(t, x, m=1, L=1, g=9.81, forcing=0):
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
        forcing = lambda t, x: const_force
    moment_of_inertia = m * L**2
    return (x[1], (-m * g * np.sin(x[0]) + forcing(t, x)) / moment_of_inertia)


def run(
    seed: float,
    /,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_data(
        nonlinear_pendulum, 2, seed, **sim_params
    )
    input_features = ["x", "x'"]
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)


    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coeff_true = [
        {"x'": 1},
        {"sin(1 x)": -9.81},
    ]
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
        plot_training_data(x_train[-1], x_train_true[-1], smoothed_last_train)
        plot_test_trajectories(x_test[-1], model, dt)

    # calculate metrics
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    return metrics


if __name__ == "__main__":
    run(seed=1, diff_params={"kind": "FiniteDifference"}, opt_params={"kind": "stlsq"})

sim_params = {"test": {"n_trajectories": 2}}
diff_params = {"test": {"kind": "FiniteDifference"}, "test2": {"kind": "SmoothedFiniteDifference"}}
feat_params = {"test": {"kind": "Polynomial"}}
opt_params = {"test": {"kind": "STLSQ"}}
