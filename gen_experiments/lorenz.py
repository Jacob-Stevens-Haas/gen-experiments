from math import ceil

import numpy as np
import pysindy as ps

from .utils import gen_data, compare_coefficient_plots, opt_lookup, diff_lookup, coeff_metrics, integration_metrics

name = "LORENZ"


def run(seed, sim_params={}, diff_params={}, opt_params={}, display=True):
    dt, t_train, x_train, x_test, x_dot_test = gen_data(ps.utils.lorenz, 3, seed, **sim_params)
    diff_cls = diff_lookup(diff_params.pop("kind"))
    diff = diff_cls(**diff_params)
    opt_cls = opt_lookup(opt_params.pop("kind"))
    opt = opt_cls(**opt_params)
    input_features = ["x", "y", "z"]

    model = ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coefficients = model.coefficients()
    coeff_true = np.array([
        [0, -10, 10, 0, 0, 0, 0, 0, 0, 0],
        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0]
    ])

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
opt_params = {"test": {"kind": "STLSQ"}}
