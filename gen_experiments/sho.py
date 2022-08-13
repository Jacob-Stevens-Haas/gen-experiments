import numpy as np
import pysindy as ps
import sklearn

from .utils import (
    gen_data,
    compare_coefficient_plots,
    opt_lookup,
    feature_lookup,
    diff_lookup,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices
)

name = "SHO"


def run(
    seed: float,
    /,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    dt, t_train, x_train, x_test, x_dot_test = gen_data(
        ps.utils.linear_damped_SHO, 2, seed, **sim_params
    )
    diff_cls = diff_lookup(diff_params.pop("kind"))
    diff = diff_cls(**diff_params)
    feature_cls = feature_lookup(feat_params.pop("kind"))
    features = feature_cls(**feat_params)
    opt_cls = opt_lookup(opt_params.pop("kind"))
    opt = opt_cls(**opt_params)
    input_features = ["x", "y"]

    model = ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,
        feature_library=features,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coefficients = model.coefficients()
    coeff_true = [
        {"x": -.1, "y": 2},
        {"x": -2, "y": -.1},
    ]
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    # make the plots
    if display:
        model.print()
        feature_names = model.get_feature_names()
        compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )

    # calculate metrics
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    return metrics


if __name__ == "__main__":
    run(seed=1, diff_params={"kind": "FiniteDifference"}, opt_params={"kind": "stlsq"})

sim_params = {"test": {"n_trajectories": 2}}
diff_params = {"test": {"kind": "FiniteDifference"}}
feat_params = {"test": {"kind": "Polynomial"}}
opt_params = {"test": {"kind": "STLSQ"}}
