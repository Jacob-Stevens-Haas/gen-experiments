from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from types import ModuleType
from typing import Annotated, Any, Sequence, Mapping, Optional, Collection, Callable, TypedDict
from math import ceil
from warnings import warn

import matplotlib.pyplot as plt
import kalman
import numpy as np
import seaborn as sns
import scipy
import pysindy as ps
import sklearn
import auto_ks as aks

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
PAL = sns.color_palette("Set1")
PLOT_KWS = dict(alpha=0.7, linewidth=3)
TRIALS_FOLDER = Path(__file__).parent.absolute() / "trials"


def gen_data(
    rhs_func,
    n_coord,
    seed=None,
    n_trajectories=1,
    x0_center=None,
    ic_stdev=3,
    noise_abs=None,
    noise_rel=None,
    nonnegative=False,
    dt=0.01,
    t_end=10,
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
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise-to-signal power ratio.
            Either noise_abs or noise_rel must be None.  Defaults to
            None.
        nonnegative (bool): Whether x0 must be nonnegative, such as for
            population models.  If so, a gamma distribution is
            used, rather than a normal distribution.

    Returns:
        dt, t_train, x_train, x_test, x_dot_test, x_train_true
    """
    if noise_abs is not None and noise_rel is not None:
        raise ValueError("Cannot specify both noise_abs and noise_rel")
    elif noise_abs is None and noise_rel is None:
        noise_abs = 0.1
    rng = np.random.default_rng(seed)
    if x0_center is None:
        x0_center = np.zeros((n_coord))
    t_train = np.arange(0, t_end, dt)
    t_train_span = (t_train[0], t_train[-1])
    if nonnegative:
        shape = ((x0_center + 1) / ic_stdev) ** 2
        scale = ic_stdev**2 / (x0_center + 1)
        x0_train = np.array(
            [rng.gamma(k, theta, n_trajectories) for k, theta in zip(shape, scale)]
        ).T
        x0_test = np.array(
            [
                rng.gamma(k, theta, ceil(n_trajectories / 2))
                for k, theta in zip(shape, scale)
            ]
        ).T
    else:
        x0_train = ic_stdev * rng.standard_normal((n_trajectories, n_coord)) + x0_center
        x0_test = (
            ic_stdev * rng.standard_normal((ceil(n_trajectories / 2), n_coord))
            + x0_center
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

    def _drop_and_warn(arrs):
        maxlen = max(arr.shape[0] for arr in arrs)

        def _alert_short(arr):
            if arr.shape[0] < maxlen:
                warn(message="Dropping simulation due to blow-up")
                return False
            return True

        arrs = list(filter(_alert_short, arrs))
        if len(arrs) == 0:
            raise ValueError(
                "Simulations failed due to blow-up.  System is too stiff for solver's"
                " numerical tolerance"
            )
        return arrs

    x_train = _drop_and_warn(x_train)
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
    x_test = _drop_and_warn(x_test)
    x_test = np.array(x_test)
    x_dot_test = np.array([[rhs_func(0, xij) for xij in xi] for xi in x_test])
    x_train_true = np.copy(x_train)
    if noise_rel is not None:
        noise_abs = np.sqrt(_signal_avg_power(x_test) * noise_rel)
    x_train = x_train + noise_abs * rng.standard_normal(x_train.shape)
    x_train = [xi for xi in x_train]
    x_test = [xi for xi in x_test]
    x_dot_test = [xi for xi in x_dot_test]
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true

def gen_pde_data(
    rhs_func: Callable,
    init_cond: np.ndarray,
    args: tuple,
    dimension: int,
    seed: int | None = None,
    noise_abs: float | None = None,
    noise_rel: float | None = None,
    dt: float = 0.01,
    t_end: int = 100,
):
    """Generate PDE measurement data for training

    For simplicity, Trajectories have been removed,
    Test data is the same as Train data.

    Arguments:
        rhs_func: the function to integrate
        init_cond: Initial Conditions for the PDE
        args: Arguments for rhsfunc 
        seed (int): the random seed for number generation
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise relative to amplitude of
            true data.  Amplitude of data is calculated as the max value
             of the power spectrum.  Either noise_abs or noise_rel must
             be None.  Defaults to None.
        dt (float): time step for the PDE simulation
        t_end (int): total time for the PDE simulation

    Returns:
        dt, t_train, x_train, x_test, x_dot_test, x_train_true
    """
    if noise_abs is not None and noise_rel is not None:
        raise ValueError("Cannot specify both noise_abs and noise_rel")
    elif noise_abs is None and noise_rel is None:
        noise_abs = .1
    rng = np.random.default_rng(seed)
    t_train = np.arange(0, t_end, dt)
    t_train_span = (t_train[0], t_train[-1])
    x_train = []
    x_train.append(
            scipy.integrate.solve_ivp(
            rhs_func,
            t_train_span,
            init_cond,
            t_eval=t_train,
            args=args, 
            **INTEGRATOR_KEYWORDS
        ).y.T
    )
    t, x = x_train[0].shape
    x_train = np.stack(x_train, axis=-1)
    if dimension==1:
        pass
    elif dimension==2:
        x_train = np.reshape(x_train, (t, int(np.sqrt(x)), int(np.sqrt(x)), 1))
    elif dimension==3:
        x_train = np.reshape(x_train, (t, int(np.cbrt(x)), int(np.cbrt(x)), int(np.cbrt(x)), 1))
    x_test = x_train
    x_test = np.moveaxis(x_test, -1, 0)
    x_dot_test = np.array(
        [[rhs_func(0, xij, args[0], args[1]) for xij in xi] for xi in x_test]
    )
    if dimension==1:
        x_dot_test = [np.moveaxis(x_dot_test, [0, 1], [-1, -2])]
        pass
    elif dimension==2:
        x_dot_test = np.reshape(x_dot_test, (t, int(np.sqrt(x)), int(np.sqrt(x)), 1))
        x_dot_test = [np.moveaxis(x_dot_test, 0, -2)]
    elif dimension==3:
        x_dot_test = np.reshape(x_dot_test, (t, int(np.cbrt(x)), int(np.cbrt(x)), int(np.cbrt(x)), 1))
        x_dot_test = [np.moveaxis(x_dot_test, 0, -2)]
    x_train_true = np.copy(x_train)
    if noise_rel is not None:
        noise_abs = _max_amplitude(x_test) * noise_rel
    x_train = x_train + noise_abs * rng.standard_normal(x_train.shape)
    x_train = [np.moveaxis(x_train, 0, -2)]
    x_train_true = np.moveaxis(x_train_true, 0, -2)
    x_test = [np.moveaxis(x_test, [0, 1], [-1, -2])]
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true


def _max_amplitude(signal: np.ndarray):
    return np.abs(scipy.fft.rfft(signal, axis=0)[1:]).max() / np.sqrt(len(signal))


def _signal_avg_power(signal: np.ndarray) -> float:
    return np.square(signal).mean()


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
    elif normalized_kind == "weak":
        return ps.WeakPDELibrary
    elif normalized_kind == "pde":
        return ps.PDELibrary
    else:
        raise ValueError


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    elif normalized_kind == "miosr":
        return ps.MIOSR
    elif normalized_kind == "ensemble":
        return ps.EnsembleOptimizer
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
    metrics["mse-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_squared_error,
    )
    metrics["mae-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_absolute_error,
    )
    return metrics


def unionize_coeff_matrices(
    model: ps.SINDy, coeff_true: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reformat true coefficients and coefficient matrix compatibly

    In order to calculate accuracy metrics between true and estimated
    coefficients, this function compares the names of true coefficients
    and a the fitted model's features in order to create comparable
    (i.e. non-ragged) true and estimated coefficient matrices.  In
    a word, it stacks the correct coefficient matrix and the estimated
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
    input_features: list[str],
    dt: float,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
) -> ps.SINDy:
    """Build a model with object parameters dictionaries

    e.g. {"kind": "finitedifference"} instead of FiniteDifference()
    """

    def finalize_param(lookup_func, pdict, lookup_key):
        try:
            cls_name = pdict.pop(lookup_key)
        except AttributeError:
            cls_name = pdict.vals.pop(lookup_key)
            pdict = pdict.vals

        param_cls = lookup_func(cls_name)
        param_final = param_cls(**pdict)
        pdict[lookup_key] = cls_name
        return param_final

    diff = finalize_param(diff_lookup, diff_params, "diffcls")
    features = finalize_param(feature_lookup, feat_params, "featcls")
    opt = finalize_param(opt_lookup, opt_params, "optcls")
    return ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,
        feature_library=features,
        feature_names=input_features,
    )


def plot_training_trajectory(
    ax: plt.Axes, x_train: np.ndarray, x_true: np.ndarray, x_smooth: np.ndarray
) -> None:
    """Plot a single training trajectory"""
    if x_train.shape[1] == 2:
        ax.plot(
            x_true[:, 0], x_true[:, 1], ".", label= "True", color=PAL[0], **PLOT_KWS
        )
        ax.plot(
            x_train[:, 0], x_train[:, 1], ".", label="Measured", color=PAL[1], **PLOT_KWS,
        )
        if (
            np.linalg.norm(x_smooth - x_train) / x_smooth.size
            > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                ".",
                label="Smoothed",
                color=PAL[2],
                **PLOT_KWS,
            )
        ax.set(xlabel="$x_0$", ylabel="$x_1$")
    elif x_train.shape[1] == 3:
        ax.plot(
            x_true[:, 0],
            x_true[:, 1],
            x_true[:, 2],
            color=PAL[0],
            label="True values",
            **PLOT_KWS,
        )

        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            x_train[:, 2],
            ".",
            color=PAL[1],
            label="Measured values",
            alpha=0.3,
        )
        if (
            np.linalg.norm(x_smooth - x_train) / x_smooth.size
            > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                x_smooth[:, 2],
                ".",
                color=PAL[2],
                label="Smoothed values",
                alpha=0.3,
            )
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    else:
        raise ValueError("Can only plot 2d or 3d data.")


def plot_training_data(
    x_train: np.ndarray, x_true: np.ndarray, x_smooth: np.ndarray
):
    """Plot training data (and smoothed training data, if different)."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_training_trajectory(axs[0], x_train, x_true, x_smooth)
    axs[0].legend()
    axs[0].set(title="Training data")
    axs[1].loglog(np.abs(scipy.fft.rfft(x_train, axis=0)) / np.sqrt(len(x_train)))
    axs[1].set(title="Training Data Absolute Spectral Density")
    axs[1].set(xlabel="Wavenumber")
    axs[1].set(ylabel="Magnitude")
    return fig


def plot_pde_training_data(last_train, last_train_true, smoothed_last_train):
    """Plot training data (and smoothed training data, if different)."""
    #1D:
    if len(last_train.shape)==3:
        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        axs[0].imshow(last_train_true, vmin=0, vmax=last_train_true.max())
        axs[0].set(title="True Data")
        axs[1].imshow(last_train_true - last_train, vmin=0, 
                    vmax=last_train_true.max())
        axs[1].set(title="Noise")
        axs[2].imshow(last_train_true - smoothed_last_train, vmin=0, 
                    vmax=last_train_true.max())
        axs[2].set(title="Smoothed Data")
        return plt.show()


def plot_test_sim_data_1d_panel(
    axs: Sequence[plt.Axes],
    x_test: np.ndarray,
    x_sim: np.ndarray,
    t_test: np.ndarray,
    t_sim: np.ndarray
) -> None:
    for ordinate, ax in enumerate(axs):
        ax.plot(t_test, x_test[:, ordinate], "k", label="true trajectory")
        axs[ordinate].plot(t_sim, x_sim[:, ordinate], "r--", label="model simulation")
        axs[ordinate].legend()
        axs[ordinate].set(xlabel="t", ylabel="$x_{}$".format(ordinate))


def _plot_test_sim_data_2d(
    axs: Annotated[Sequence[plt.Axes], "len=2"], x_test: np.ndarray, x_sim: np.ndarray
) -> None:
    axs[0].plot(x_test[:, 0], x_test[:, 1], "k")
    axs[0].set(xlabel="$x_0$", ylabel="$x_1$")
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], "r--")
    axs[1].set(xlabel="$x_0$", ylabel="$x_1$")


def _plot_test_sim_data_3d(
    axs: Annotated[Sequence[plt.Axes], "len=3"], x_test: np.ndarray, x_sim: np.ndarray
) -> None:
    axs[0].plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], "k")
    axs[0].set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], "r--")
    axs[1].set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")


def plot_test_trajectories(
    x_test: np.ndarray, model: ps.SINDy, dt: float
) -> Mapping[str, np.ndarray]:
    """Plot a test trajectory

    Args:
        last_test: a single trajectory of the system
        model: a trained model to simulate and compare to test data
        dt: the time interval in test data

    Returns:
        A dict with two keys, "t_sim" (the simulation times) and
    "x_sim" (the simulated trajectory)
    """
    t_test = np.arange(len(x_test) * dt, step=dt)
    t_sim = t_test
    try:
        x_sim = model.simulate(x_test[0], t_test)
    except ValueError:
        warn(message="Simulation blew up; returning zeros")
        x_sim = np.zeros_like(x_test)
    # truncate if integration returns wrong number of points
    t_sim = t_test[: len(x_sim)]
    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    plt.suptitle("Test Trajectories by Dimension")
    plot_test_sim_data_1d_panel(axs, x_test, x_sim, t_test, t_sim)
    axs[-1].legend()

    plt.suptitle("Full Test Trajectories")
    if x_test.shape[1] == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        _plot_test_sim_data_2d(axs, x_test, x_sim)
    elif x_test.shape[1] == 3:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={"projection": "3d"})
        _plot_test_sim_data_3d(axs, x_test, x_sim)
    else:
        raise ValueError("Can only plot 2d or 3d data.")
    axs[0].set(title="true trajectory")
    axs[1].set(title="model simulation")
    return {"t_sim": t_sim, "x_sim": x_sim}


@dataclass
class ParamDetails:
    vals: Sequence | Mapping
    modules: Sequence[ModuleType]


@dataclass
class SeriesDef:
    """The details of constructing the ragged axes of a grid search.

    The concept of a SeriesDef refers to a slice along a single axis of
    a grid search in conjunction with another axis (or axes)
    whose size or meaning differs along different slices.

    Attributes:
        name: The name of the slice, as a label for printing
        static_param: the constant paramter to this slice. Then key is
            the name of the parameter, as understood by the experiment
            Conceptually, the key serves as an index of this slice in
            the gridsearch.
        grid_params: the keys of the parameters in the experiment that
            vary along jagged axis for this slice
        grid_vals: the values of the parameters in the experiment that
            vary along jagged axis for this slice

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )

    """
    name: str
    static_param: dict
    grid_params: Optional[Sequence[str]]
    grid_vals: Optional[list[Sequence]]


@dataclass
class SeriesList:
    """Specify the ragged slices of a grid search.

    As an example, consider a grid search of miles per gallon for
    different vehicles, in different routes, with different tires.
    Since different tires fit on different vehicles, the tire axis would
    be ragged, varying along the vehicle axis.

        Truck = SeriesDef("trucks")

    Attribtues:
        param_name: the key of the parameter in the experiment that
            varies along the series axis.
        print_name: the print name of the parameter in the experiment
            that varies along the series axis.
        series_list: Each element of the series axis

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )
        bike_tires = SeriesDef(
            "Bike",
            {"vehicle": "bicycle"},
            ["vehicle.tires"],
            [["gravel_tires", "road_tires"]]
        )
        VehicleOptions = SeriesList(
            "vehicle",
            "Vehicle Types",
            [truck_wheels, bike_tires]
        )

    """
    param_name: Optional[str]
    print_name: Optional[str]
    series_list: list[SeriesDef]


class NestedDict(defaultdict):
    """A dictionary that splits all keys by ".", creating a sub-dict.

    Args: see superclass

    Example:

        >>> foo = NestedDict("a.b"=1)
        >>> foo["a.c"] = 2
        >>> foo["a"]["b"]
        1
    """
    def __missing__(self, key):
        try:
            prefix, subkey = key.split(".", 1)
        except ValueError:
            raise KeyError(key)
        return self[prefix][subkey]

    def __setitem__(self, key, value):
        if "." in key:
            prefix, suffix = key.split(".", 1)
            if self.get(prefix) is None:
                self[prefix] = NestedDict()
            return self[prefix].__setitem__(suffix, value)
        else:
            return super().__setitem__(key, value)

    def update(self, other: dict):
        try:
            for k, v in other.items():
                self.__setitem__(k, v)
        except:
            super().update(other)


@dataclass(frozen=True)
class _PlotPrefs:
    plot: bool = True
    rel_noise: bool = False
    grid_plot_match: Collection[dict] = field(default_factory=lambda: (dict(),))

    def __bool__(self):
        return self.plot


def kalman_generalized_cv(
    times: np.ndarray, measurements: np.ndarray, alpha0: float = 1, detail=False
):
    """Find kalman parameter alpha using GCV error

    See Boyd & Barratt, Fitting a Kalman Smoother to Data.  No regularization
    """
    measurements = measurements.reshape((-1, 1))
    nt = len(measurements)
    dt = times[1] - times[0]
    Ai = np.array([[1, 0], [dt, 1]])
    Qi = kalman.gen_Qi(dt)
    Qi_rt_inv = np.linalg.cholesky(np.linalg.inv(Qi))
    Qi_r_i_vec = np.reshape(Qi_rt_inv, (-1, 1))
    Qi_proj = (
        lambda vec: Qi_r_i_vec
        @ (Qi_r_i_vec.T @ Qi_r_i_vec) ** -1
        @ (Qi_r_i_vec.T)
        @ vec
    )
    Hi = np.array([[0, 1]])
    Ri = np.eye(1)
    Ri_rt_inv = Ri
    params0 = aks.KalmanSmootherParameters(Ai, Qi_rt_inv, Hi, Ri)
    mask = np.ones_like(measurements, dtype=bool)
    mask[::4] = False

    def proj(curr_params, t):
        W_n_s_v = np.reshape(curr_params.W_neg_sqrt, (-1, 1))
        W_n_s_v = np.reshape(Qi_proj(W_n_s_v), (2, 2))
        new_params = aks.KalmanSmootherParameters(Ai, W_n_s_v, Hi, Ri_rt_inv)
        return new_params, t

    params, info = aks.tune(params0, proj, measurements, K=mask, lam=0.1, verbose=False)
    est_Q = np.linalg.inv(params.W_neg_sqrt @ params.W_neg_sqrt.T)
    est_alpha = 1 / (est_Q / Qi).mean()
    return est_alpha


class GridPointData(TypedDict):
    x_train: np.ndarray
    x_true: np.ndarray
    smooth_train: np.ndarray
    x_test: np.ndarray
    t_sim: np.ndarray
    x_sim: np.ndarray


class PlotData(TypedDict):
    params: dict[str, Any]
    data: GridPointData


SeriesData = Annotated[
    list[Annotated[np.ndarray, "(n_metrics, n_grid_vals)"]], "len=n_grid_axes"
]

class Results(TypedDict):
    plot_data: list[PlotData]
    series_data: dict[str, SeriesData]
    metrics: list[str]
    grid_axes: dict[str, Collection[float]]
    main: float


def load_results(hexstr: str) -> Results:
    """Load the results that mitosis saves

    Args:
        hexstr: randomly-assigned identifier for the results to open
    """
    with open(TRIALS_FOLDER / f"results_{hexstr}.npy", "rb") as f:
        return np.load(f, allow_pickle=True)[()]


def _setup_summary_fig(n_subplots) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    n_rows = max(n_subplots // 3, (n_subplots + 2) // 3)
    fig, subplots = plt.subplots(
        n_rows,
        min(n_subplots, 3),
        sharey="row",
        sharex="col",
        squeeze=False,
    )
    return fig, subplots


def plot_summary_smoothing(params: dict, *args: tuple[str, str]) -> None:
    """Plot a single parameter's smoothing/training across multiple ODEs
    """
    fig, axs = _setup_summary_fig(len(args))
    fig.suptitle("How well does a smoothing method perform across ODEs?")

    for ax, (ode_name, hexstr) in zip(axs.reshape(-1), args):
        results = load_results(hexstr)
        for trajectory in results["plot_data"]:
            if params == trajectory["params"]:
                ax.set_title(ode_name)
                plot_training_trajectory(
                    ax,
                    trajectory["data"]["x_train"],
                    trajectory["data"]["x_true"],
                    trajectory["data"]["smooth_train"]
                )
                break

        else:
            warn(f"Did not find a parameter match for {ode_name} experiment")
    ax.legend()


def plot_summary_simulation(params: dict, *args: tuple[str, str]) -> None:
    """Plot a single parameter's smoothing/training across multiple ODEs
    """
    fig, axs = _setup_summary_fig(len(args))
    fig.suptitle("How well does a smoothing method perform across ODEs?")

    for ax, (ode_name, hexstr) in zip(axs.reshape(-1), args):
        results = load_results(hexstr)
        for trajectory in results["plot_data"]:
            if params == trajectory["params"]:
                ax.set_title(ode_name)
                plot_training_trajectory(
                    ax,
                    trajectory["data"]["x_train"],
                    trajectory["data"]["x_true"],
                    trajectory["data"]["smooth_train"]
                )
                break

        else:
            warn(f"Did not find a parameter match for {ode_name} experiment")
    ax.legend()


def plot_summary_metric(
    metric: str, grid_axis_name: tuple[str, Collection], *args: tuple[str, str]
) -> None:
    """After multiple gridsearches, plot a comparison for all ODEs

    Plots the overall results for a single metric, single grid axis
    Args:
        metric: which metric is being plotted
        grid_axis: the name of the parameter varied and the values of
            the parameter.
        *args: each additional tuple contains the name of an ODE and
            the hexstr under which it's data is saved.
    """
    fig, axs = _setup_summary_fig(len(args))
    fig.suptitle(
        f"How well do the methods work on different ODEs as {grid_axis_name} changes?"
    )
    for ax, (ode_name, hexstr) in zip(axs.reshape(-1), args):
        results = load_results(hexstr)
        grid_axis_index = results["grid_params"].index(grid_axis_name)
        grid_axis = results["grid_vals"][grid_axis_index]
        metric_index = results["metrics"].index(metric)
        for s_name, s_data in results["series_data"].items():
            ax.plot(grid_axis, s_data[grid_axis_index][metric_index], label=s_name)
        ax.set_title(ode_name)
    ax.legend()
