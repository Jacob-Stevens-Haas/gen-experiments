import numpy as np
import pysindy as ps

from .utils import (
    gen_data,
    compare_coefficient_plots,
    plot_training_data,
    plot_test_trajectories,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model,
)

name = "LORENZ"


def run(
    seed: float,
    /,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    x0_center = np.array([0, 0, 15])
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_data(
        ps.utils.lorenz, 3, seed, x0_center=x0_center, **sim_params
    )
    input_features = ["x", "y", "z"]
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coeff_true = [
        {"x": -10, "y": 10},
        {"x": 28, "y": -1, "x z": -1},
        {"z": -8 / 3, "x y": 1},
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
    run(
        seed=1,
        diff_params={"diffcls": "FiniteDifference"},
        opt_params={"optcls": "stlsq"},
    )
