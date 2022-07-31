from math import ceil

import numpy as np
import pysindy as ps
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}


def gen_data(seed=None, example_trajectory=False):
    """Generate random training and test data"""
    rng = np.random.default_rng(seed)
    dt = 0.01
    ic_stdev = 3
    noise_stdev = 0.1
    n_trajectories = 1
    t_train = np.arange(0, 10, dt)
    t_train_span = (t_train[0], t_train[-1])
    if example_trajectory:
        x0_train = np.array([[2, 0]])
        x0_test = np.array([[2, 0]])
    else:
        x0_train = ic_stdev * rng.standard_normal((n_trajectories, 2))
        x0_test = ic_stdev * rng.standard_normal((ceil(n_trajectories / 2), 2))
    x_train = []
    for traj in range(n_trajectories):
        x_train.append(
            scipy.integrate.solve_ivp(
                ps.utils.linear_damped_SHO,
                t_train_span,
                x0_train[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS
            ).y.T
        )
    x_train = np.stack(x_train)
    x_test = []
    for traj in range(ceil(n_trajectories / 2)):
        x_test.append(
            scipy.integrate.solve_ivp(
                ps.utils.linear_damped_SHO,
                t_train_span,
                x0_test[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS
            ).y.T
        )
    x_test = np.array(x_test)
    x_dot_test = np.array(
        [[ps.utils.linear_damped_SHO(0, xij) for xij in xi] for xi in x_test]
    )
    x_train = x_train + noise_stdev * rng.standard_normal(x_train.shape)
    x_train = [xi for xi in x_train]
    x_test = [xi for xi in x_test]
    x_dot_test = [xi for xi in x_dot_test]
    return dt, t_train, x_train, x_test, x_dot_test


def diff_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "finitedifference":
        return ps.FiniteDifference
    elif normalized_kind == "sindy":
        return ps.SINDyDerivative
    else:
        raise ValueError


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    else:
        raise ValueError


def run(seed, sim_params={}, diff_params={}, opt_params={}):
    dt, t_train, x_train, x_test, x_dot_test = gen_data(seed, **sim_params)
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
    coeff_true = np.array([
        [0, -.1, 2, 0, 0, 0],
        [0, -2, -.1, 0, 0, 0]
    ])

    # make the plots
    model.print()
    input_features = ["x", "y"]
    feature_names = model.get_feature_names()
    compare_coefficient_plots(
        coefficients, coeff_true, input_features=input_features, feature_names=feature_names
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
    return metrics

def plot_coefficients(
    coefficients, input_features=None, feature_names=None, ax=None, **heatmap_kws
):
    if input_features is None:
        input_features = [r"$\dot x_" + f"{k}$" for k in range(coefficients.shape[0])]
    else:
        input_features = [r"$\dot " + f"{fi}$" for fi in input_features]

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(coefficients.shape[1])]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        heatmap_args = {
            "xticklabels": input_features,
            "yticklabels": feature_names,
            "center": 0.0,
            "cmap": sns.color_palette("vlag", n_colors=20, as_cmap=True),
            "ax": ax,
            "linewidths": 0.1,
            "linecolor": "whitesmoke",
        }
        heatmap_args.update(**heatmap_kws)

        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax

def compare_coefficient_plots(
    coefficients_est, coefficients_true, input_features=None, feature_names=None
):
    n_cols = len(coefficients_est)

    def signed_sqrt(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        fig, axs = plt.subplots(
            1, 2, figsize=(1.9 * n_cols, 8), sharey=True, sharex=True
        )

        max_clean = max(np.max(np.abs(c)) for c in coefficients_est)
        max_noisy = max(np.max(np.abs(c)) for c in coefficients_true)
        max_mag = np.sqrt(max(max_clean, max_noisy))

        plot_coefficients(
            signed_sqrt(coefficients_true),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[0],
            cbar=False,
            vmax=max_mag,
            vmin=-max_mag,
        )

        plot_coefficients(
            signed_sqrt(coefficients_est),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[1],
            cbar=False,
        )

        axs[0].set_title("True Coefficients", rotation=45)
        axs[1].set_title("Est. Coefficients", rotation=45)

        fig.tight_layout()

if __name__ == "__main__":
    run(seed=1, diff_params={"kind": "FiniteDifference"}, opt_params={"kind": "stlsq"})