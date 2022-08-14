from itertools import chain
from multiprocessing.sharedctypes import Value
from typing import List, Tuple
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import pysindy as ps
import sklearn

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
PAL = sns.color_palette("Set1")
PLOT_KWS = dict(alpha=0.7, linewidth=3)


def gen_data(
    rhs_func,
    n_coord,
    seed=None,
    n_trajectories=1,
    x0_center=None,
    ic_stdev=3,
    noise_stdev=0.1,
):
    """Generate random training and test data

    Note that test data has no noise.

    Arguments:
        rhs_func (Callable): the function to integrate
        n_coord (int): number of coordinates needed for rhs_func
        seed (int): the random seed for number generation
        n_trajectories (int): number of trajectories of training data
        x0_center (np.array): center of random initial conditions
        ic_stdev (float): standard deviation for generating initial
            conditions
        noise_stdev (float): measurement noise standard deviation
    """
    rng = np.random.default_rng(seed)
    if x0_center is None:
        x0_center = np.zeros((n_coord))
    dt = 0.01
    t_train = np.arange(0, 10, dt)
    t_train_span = (t_train[0], t_train[-1])
    x0_train = ic_stdev * rng.standard_normal((n_trajectories, n_coord)) + x0_center
    x0_test = (
        ic_stdev * rng.standard_normal((ceil(n_trajectories / 2), n_coord)) + x0_center
    )
    x_train = []
    for traj in range(n_trajectories):
        x_train.append(
            scipy.integrate.solve_ivp(
                rhs_func,
                t_train_span,
                x0_train[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS,
            ).y.T
        )
    x_train = np.stack(x_train)
    x_test = []
    for traj in range(ceil(n_trajectories / 2)):
        x_test.append(
            scipy.integrate.solve_ivp(
                rhs_func,
                t_train_span,
                x0_test[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS,
            ).y.T
        )
    x_test = np.array(x_test)
    x_dot_test = np.array([[rhs_func(0, xij) for xij in xi] for xi in x_test])
    x_train_true = np.copy(x_train)
    x_train = x_train + noise_stdev * rng.standard_normal(x_train.shape)
    x_train = [xi for xi in x_train]
    x_test = [xi for xi in x_test]
    x_dot_test = [xi for xi in x_dot_test]
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true


def diff_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "finitedifference":
        return ps.FiniteDifference
    if normalized_kind == "smoothedfinitedifference":
        return ps.SmoothedFiniteDifference
    elif normalized_kind == "sindy":
        return ps.SINDyDerivative
    else:
        raise ValueError


def feature_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind is None:
        return ps.PolynomialLibrary
    elif normalized_kind == "polynomial":
        return ps.PolynomialLibrary
    elif normalized_kind == "fourier":
        return ps.FourierLibrary


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    else:
        raise ValueError


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


def coeff_metrics(coefficients, coeff_true):
    metrics = {}
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


def integration_metrics(model, x_test, t_train, x_dot_test):
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
    return metrics


def unionize_coeff_matrices(
    model: ps.SINDy, coeff_true: List[dict]
) -> Tuple[np.ndarray]:
    """Reformat true coefficients and coefficient matrix compatibly


    In order to calculate accuracy metrics between true and estimated
    coefficients, this function compares the names of true coefficients
    and a the fitted model's features in order to create comparable true
    and estimated coefficient matrices.

    That is, it embeds the correct coefficient matrix and the estimated
    coefficient matrix in a matrix that represents the union of true
    features and modeled features.

    Arguments:
        model: fitted model
        coeff_true: list of dicts of format function_name: coefficient,
            one dict for each modeled coordinate/target

    Returns:
        Tuple of true coefficient matrix, estimated coefficient matrix,
        and combined feature names

    Warning:
        Does not disambiguate between commutatively equivalent function
        names such as 'x z' and 'z x' or 'x^2' and 'x x'
    """
    model_features = model.get_feature_names()
    true_features = [set(coeffs.keys()) for coeffs in coeff_true]
    unmodeled_features = set(chain.from_iterable(true_features)) - set(model_features)
    model_features.extend(list(unmodeled_features))
    est_coeff_mat = model.coefficients()
    new_est_coeff = np.zeros((est_coeff_mat.shape[0], len(model_features)))
    new_est_coeff[:, : est_coeff_mat.shape[1]] = est_coeff_mat
    true_coeff_mat = np.zeros_like(new_est_coeff)
    for row, terms in enumerate(coeff_true):
        for term, coeff in terms.items():
            true_coeff_mat[row, model_features.index(term)] = coeff

    return true_coeff_mat, new_est_coeff, model_features


def _make_model(
    input_features: List[str],
    dt: float,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
) -> ps.SINDy:
    """Build a model with object parameters dictionaries

    e.g. {"kind": "finitedifference"} instead of FiniteDifference()
    """
    diff_cls = diff_lookup(diff_params.pop("kind"))
    diff = diff_cls(**diff_params)
    feature_cls = feature_lookup(feat_params.pop("kind"))
    features = feature_cls(**feat_params)
    opt_cls = opt_lookup(opt_params.pop("kind"))
    opt = opt_cls(**opt_params)
    return ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,
        feature_library=features,
        feature_names=input_features,
    )


def plot_training_data(last_train, last_train_true, smoothed_last_train):
    """Plot training data (and smoothed training data, if different)."""
    plt.figure(figsize=[6, 6])
    ax = plt.gca()
    if last_train.shape[1] == 2:
        ax.plot(
            last_train_true[:, 0],
            last_train_true[:, 1],
            ".",
            label="True values",
            color=PAL[0],
            **PLOT_KWS,
        )
        ax.plot(
            last_train[:, 0],
            last_train[:, 1],
            ".",
            label="Measured values",
            color=PAL[1],
            **PLOT_KWS,
        )
        if (
            np.linalg.norm(smoothed_last_train - last_train) / smoothed_last_train.size
            > 1e-12
        ):
            ax.plot(
                smoothed_last_train[:, 0],
                smoothed_last_train[:, 1],
                ".",
                label="Smoothed values",
                color=PAL[2],
                **PLOT_KWS,
            )
        ax.set(xlabel="$x_0$", ylabel="$x_1$")
    elif last_train.shape[1] == 3:
        ax = plt.axes(projection="3d")
        ax.plot(
            last_train_true[:, 0],
            last_train_true[:, 1],
            last_train_true[:, 2],
            color=PAL[0],
            label="True values",
            **PLOT_KWS,
        )

        ax.plot(
            last_train[:, 0],
            last_train[:, 1],
            last_train[:, 2],
            ".",
            color=PAL[1],
            label="Measured values",
            alpha=0.3,
        )
        if (
            np.linalg.norm(smoothed_last_train - last_train) / smoothed_last_train.size
            > 1e-12
        ):
            ax.plot(
                smoothed_last_train[:, 0],
                smoothed_last_train[:, 1],
                smoothed_last_train[:, 2],
                ".",
                color=PAL[2],
                label="Smoothed values",
                alpha=0.3,
            )
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    else:
        raise ValueError("Can only plot 2d or 3d data.")
    ax.set(title="Training data")
    ax.legend()
    return ax


def plot_test_trajectories(last_test, model, dt):
    t_test = np.arange(len(last_test) * dt, step=dt)
    x_test_sim = model.simulate(last_test[0], t_test)
    fig, axs = plt.subplots(last_test.shape[1], 1, sharex=True, figsize=(7, 9))
    plt.suptitle("Trajectories by Dimension")
    for i in range(last_test.shape[1]):
        axs[i].plot(t_test, last_test[:, i], "k", label="true trajectory")
        axs[i].plot(t_test, x_test_sim[:, i], "r--", label="model simulation")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

    fig = plt.figure(figsize=(10, 4.5))
    plt.suptitle("Full trajectories")
    if last_test.shape[1] == 2:
        ax1 = fig.add_subplot(121)
        ax1.plot(last_test[:, 0], last_test[:, 1], "k")
        ax1.set(xlabel="$x_0$", ylabel="$x_1$", title="true trajectory")
        ax2 = fig.add_subplot(122)
        ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], "r--")
        ax2.set(xlabel="$x_0$", ylabel="$x_1$", title="model simulation")
    elif last_test.shape[1] == 3:
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(last_test[:, 0], last_test[:, 1], last_test[:, 2], "k")
        ax1.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="true trajectory")
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], "r--")
        ax2.set(
            xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="model simulation"
        )
    else:
        raise ValueError("Can only plot 2d or 3d data.")
