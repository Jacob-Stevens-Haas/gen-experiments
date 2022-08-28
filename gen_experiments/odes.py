import pysindy as ps
import numpy as np

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

name = "odes"

p_duff = [0.2, 0.05, 1]
p_lotka = [1, 10]
p_ross = [0.2, 0.2, 5.7]
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
        "rhsfunc": ps.utils.odes.lotka,
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
            {
                "y": -1,
                "z": -1,
            },
            {"x": 1, "y": p_ross[0]},
            {"1": p_ross[1], "z": -p_ross[2], "x z": 1},
        ],
    },
}


def run(
    seed: float,
    /,
    group: str,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    rhsfunc = ode_setup[group]["rhsfunc"]
    input_features = ode_setup[group]["input_features"]
    coeff_true = ode_setup[group]["coeff_true"]
    try:
        x0_center = ode_setup[group]["x0_center"]
    except KeyError:
        x0_center = np.zeros(len(input_features))
    try:
        nonnegative = ode_setup[group]["nonnegative"]
    except KeyError:
        nonnegative = False
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_data(
        rhsfunc,
        len(input_features),
        seed,
        x0_center=x0_center,
        nonnegative=nonnegative,
        **sim_params,
    )
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, quiet=True, multiple_trajectories=True)
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
    exp = "duffing"
    exp.run(
        seed=1,
        group_name="duffing",
        diff_params={"kind": "FiniteDifference"},
        opt_params={"kind": "stlsq"},
    )
