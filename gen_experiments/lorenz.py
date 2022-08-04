from math import ceil

import numpy as np
import pysindy as ps
import scipy
import sklearn

from .utils import gen_data, compare_coefficient_plots, opt_lookup, diff_lookup, INTEGRATOR_KEYWORDS

name = "LORENZ"


def run(seed, sim_params={}, diff_params={}, opt_params={}, display=True):
    dt, t_train, x_train, x_test, x_dot_test = gen_data(ps.utils.lorenz, seed, **sim_params)
    diff_cls = diff_lookup(diff_params.pop("kind"))
    diff = diff_cls(**diff_params)
    opt_cls = opt_lookup(opt_params.pop("kind"))
    opt = opt_cls(**opt_params)
    input_features = ["x", "y"]

    model = ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coefficients = model.coefficients()
    coeff_true = np.array([[0, -0.1, 2, 0, 0, 0], [0, -2, -0.1, 0, 0, 0]])

    # make the plots
    if display:
        model.print()
        input_features = ["x", "y"]
        feature_names = model.get_feature_names()
        compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )

    # calculate metrics
    metrics = {}
    metrics["mse"] = model.score(
        x_test, t_train, x_dot_test, multiple_trajectories=True
    )
    metrics["mae"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        multiple_trajectories=True,
        metric=sklearn.metrics.mean_absolute_error,
    )
    metrics["coeff_precision"] = sklearn.metrics.precision_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_recall"] = sklearn.metrics.recall_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_f1"] = sklearn.metrics.f1_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_mse"] = sklearn.metrics.mean_squared_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["coeff_mae"] = sklearn.metrics.mean_absolute_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["main"] = metrics["coeff_f1"]
    return metrics

if __name__ == "__main__":
    run(seed=1, diff_params={"kind": "FiniteDifference"}, opt_params={"kind": "stlsq"})

sim_params = {"test": {"n_trajectories": 2}}
diff_params = {"test": {"kind": "FiniteDifference"}}
opt_params = {"test": {"kind": "STLSQ"}}
