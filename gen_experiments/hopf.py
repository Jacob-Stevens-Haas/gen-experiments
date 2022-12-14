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

name = "hopf"


def run(
    seed: float,
    /,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    mu = -0.05
    omega = 1
    A = 1
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_data(
        ps.utils.odes.hopf, 2, seed, ic_stdev=1, **sim_params
    )
    input_features = ["x", "y"]
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coeff_true = [
        {"x": mu, "y": -omega, "x^3": -A, "x y^2": -A},
        {"x": omega, "y": mu, "x^2 y": -A, "y^3": -A},
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


sim_params = {"test": {"n_trajectories": 2}}
diff_params = {
    "test": {"kind": "FiniteDifference"},
    "test2": {"kind": "SmoothedFiniteDifference"},
}
feat_params = {"test": {"kind": "Polynomial", "degree": 3}}
opt_params = {"test": {"kind": "STLSQ"}}
