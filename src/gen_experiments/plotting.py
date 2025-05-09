from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from .gridsearch.typing import GridLocator

PAL = sns.color_palette("Set1")
PLOT_KWS = {"alpha": 0.7, "linewidth": 3}


@dataclass
class _ColorConstants:
    color_sequence: list[ColorType]

    def set_sequence(self, color_sequence: list[ColorType]):
        self.color_sequence = color_sequence

    @property
    def TRUE(self):
        return self.color_sequence[0]

    @property
    def MEAS(self):
        return self.color_sequence[1]

    @property
    def EST(self):
        return self.color_sequence[2]

    @property
    def EST2(self):
        return self.color_sequence[3]

    @property
    def TRAIN(self):
        return self.color_sequence[4]

    @property
    def TEST(self):
        return self.color_sequence[5]


COLOR = _ColorConstants(mpl.color_sequences["tab10"])


@dataclass(frozen=True)
class _PlotPrefs:
    """Control which gridsearch data gets plotted, and a bit of how

    Args:
        plot: whether to plot
        rel_noise: Whether and how to convert true noise into relative noise
        grid_params_match: dictionaries of parameters to match when plotted. OR
            is applied across the collection
        grid_ind_match: indexing tuple to match indices in a single series
            gridsearch.  Only positive integers are allowed, except the first
            element may be slice(None).  Alternatively, ellipsis to match all
            indices
    """

    plot: bool = True
    rel_noise: Literal[False] | Callable[..., dict[str, Sequence[Any]]] = False
    plot_match: GridLocator = field(default_factory=lambda: GridLocator())

    def __bool__(self):
        return self.plot


def plot_coefficients(
    coefficients: Annotated[np.ndarray, "(n_coord, n_features)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    ax: Axes,
    **heatmap_kws,
) -> None:
    """Plot a set of dynamical system coefficients in a heatmap.

    Args:
        coefficients: A 2D array holding the coefficients of different
            library functions.  System dimension is rows, function index
            is columns
        input_features: system coordinate names, e.g. "x","y","z" or "u","v"
        feature_names: the names of the functions in the library.
        ax: the matplotlib axis to plot on
        **heatmap_kws: additional kwargs to seaborn's styling
    """

    def detex(input: str) -> str:
        if input[0] == "$":
            input = input[1:]
        if input[-1] == "$":
            input = input[:-1]
        return input

    if input_features is None:
        input_features = [r"$\dot x_" + f"{k}$" for k in range(coefficients.shape[0])]
    else:
        input_features = [r"$\dot " + f"{detex(fi)}$" for fi in input_features]

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(coefficients.shape[1])]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
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
        coefficients = np.where(
            coefficients == 0, np.nan * np.empty_like(coefficients), coefficients
        )
        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax


def compare_coefficient_plots(
    coefficients_est: Annotated[np.ndarray, "(n_coord, n_feat)"],
    coefficients_true: Annotated[np.ndarray, "(n_coord, n_feat)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    scaling: bool = True,
    axs: Optional[Sequence[Axes]] = None,
):
    """Create plots of true and estimated coefficients.

    Args:
        scaling: Whether to scale coefficients so that magnitude of largest to
            smallest (in absolute value) is less than or equal to ten.
        axs: A sequence of axes of at least length two.  Plots are added to the
            first two axes in the list
    """
    n_cols = len(coefficients_est)

    # helps boost the color of small coefficients.  Maybe log is better?
    all_vals = np.hstack((coefficients_est.flatten(), coefficients_true.flatten()))
    nzs = all_vals[all_vals.nonzero()]
    max_val = np.max(np.abs(nzs), initial=0.0)
    min_val = np.min(np.abs(nzs), initial=np.inf)
    if scaling and np.isfinite(min_val) and max_val / min_val > 10:
        pwr_ratio = 1.0 / np.log10(max_val / min_val)
    else:
        pwr_ratio = 1

    def signed_root(x):
        return np.sign(x) * np.power(np.abs(x), pwr_ratio)

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        if axs is None:
            fig, axs = plt.subplots(
                1, 2, figsize=(1.9 * n_cols, 8), sharey=True, sharex=True
            )
            fig.tight_layout()

        vmax = signed_root(max_val)

        plot_coefficients(
            signed_root(coefficients_true),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[0],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        plot_coefficients(
            signed_root(coefficients_est),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[1],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        axs[0].set_title("True Coefficients", rotation=45)
        axs[1].set_title("Est. Coefficients", rotation=45)


def _plot_training_trajectory(
    ax: Axes,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: np.ndarray | None,
    labels: bool = True,
) -> None:
    """Plot a single training trajectory"""
    if x_train.shape[1] == 2:
        ax.plot(
            x_true[:, 0], x_true[:, 1], ".", label="True", color=COLOR.TRUE, **PLOT_KWS
        )
        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            ".",
            label="Measured",
            color=COLOR.MEAS,
            **PLOT_KWS,
        )
        if (
            x_smooth is not None
            and np.linalg.norm(x_smooth - x_train) / x_smooth.size > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                ".",
                label="Smoothed",
                color=COLOR.EST,
                **PLOT_KWS,
            )
        if labels:
            ax.set(xlabel="$x_0$", ylabel="$x_1$")
        else:
            ax.set(xticks=[], yticks=[])
    elif x_train.shape[1] == 3:
        ax.plot(
            x_true[:, 0],
            x_true[:, 1],
            x_true[:, 2],
            color=COLOR.TRUE,
            label="True values",
            **PLOT_KWS,
        )

        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            x_train[:, 2],
            ".",
            color=COLOR.MEAS,
            label="Measured values",
            alpha=0.3,
        )
        if (
            x_smooth is not None
            and np.linalg.norm(x_smooth - x_train) / x_smooth.size > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                x_smooth[:, 2],
                ".",
                color=COLOR.EST,
                label="Smoothed values",
                alpha=0.3,
            )
        if labels:
            ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        else:
            ax.set(xticks=[], yticks=[], zticks=[])
    else:
        raise ValueError("Can only plot 2d or 3d data.")


def plot_training_data(
    x_train: np.ndarray, x_true: np.ndarray, x_smooth: np.ndarray | None = None
) -> tuple[Figure, Figure]:
    """Plot training data (and smoothed training data, if different)."""
    fig_3d = plt.figure(figsize=(12, 6))
    if x_train.shape[-1] == 2:
        ax0 = fig_3d.add_subplot(1, 2, 1)
        coord_names = ("x", "y")
    elif x_train.shape[-1] == 3:
        ax0 = fig_3d.add_subplot(1, 2, 1, projection="3d")
        coord_names = ("x", "y", "z")
    else:
        raise ValueError("Too many or too few coordinates to plot")
    _plot_training_trajectory(ax0, x_train, x_true, x_smooth)
    ax0.legend()
    ax0.set(title="Training data")
    ax1 = fig_3d.add_subplot(1, 2, 2)
    ax1.loglog(
        np.abs(scipy.fft.rfft(x_train, axis=0)) / np.sqrt(len(x_train)),
        color=COLOR.MEAS,
    )
    ax1.loglog(
        np.abs(scipy.fft.rfft(x_true, axis=0)) / np.sqrt(len(x_true)), color=COLOR.TRUE
    )
    ax1.legend(coord_names)
    ax1.set(title="Training Data Absolute Spectral Density")
    ax1.set(xlabel="Wavenumber")
    ax1.set(ylabel="Magnitude")

    n_coord = x_true.shape[-1]
    fig_coord = plt.figure(figsize=(n_coord * 4, 6))
    for coord_ind in range(n_coord):
        ax = fig_coord.add_subplot(n_coord, 1, coord_ind + 1)
        ax.set_title(coord_names[coord_ind])
        plot_training_1d(ax, coord_ind, x_train, x_true, x_smooth)

    ax.legend()

    return fig_3d, fig_coord


def plot_training_1d(
    ax: Axes,
    coord_ind: int,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: Optional[np.ndarray],
):
    ax.plot(x_train[..., coord_ind], "b.", color=COLOR.MEAS, label="measured")
    ax.plot(x_true[..., coord_ind], "r-", color=COLOR.TRUE, label="true")
    if x_smooth is not None:
        ax.plot(x_smooth[..., coord_ind], color=COLOR.EST, label="smoothed")


def plot_pde_training_data(last_train, last_train_true, smoothed_last_train):
    """Plot training data (and smoothed training data, if different)."""
    # 1D:
    if len(last_train.shape) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(last_train_true, vmin=0, vmax=last_train_true.max())
        axs[0].set(title="True Data")
        axs[1].imshow(last_train, vmin=0, vmax=last_train_true.max())
        axs[1].set(title="Noisy Data")
        axs[2].imshow(smoothed_last_train, vmin=0, vmax=last_train_true.max())
        axs[2].set(title="Smoothed Data")
        return plt.show()


def plot_test_sim_data_1d_panel(
    axs: Sequence[Axes],
    x_test: np.ndarray,
    x_sim: np.ndarray,
    t_test: np.ndarray,
    t_sim: np.ndarray,
) -> None:
    for ordinate, ax in enumerate(axs):
        ax.plot(t_test, x_test[:, ordinate], "k", label="true trajectory")
        axs[ordinate].plot(t_sim, x_sim[:, ordinate], "r--", label="model simulation")
        axs[ordinate].legend()
        axs[ordinate].set(xlabel="t", ylabel="$x_{}$".format(ordinate))


def _plot_test_sim_data_2d(
    axs: Annotated[Sequence[Axes], "len=2"],
    x_test: np.ndarray,
    x_sim: np.ndarray,
    labels: bool = True,
) -> None:
    axs[0].plot(x_test[:, 0], x_test[:, 1], "k", label="True Trajectory")
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], "r--", label="Simulation")
    for ax in axs:
        if labels:
            ax.set(xlabel="$x_0$", ylabel="$x_1$")
        else:
            ax.set(xticks=[], yticks=[])


def _plot_test_sim_data_3d(
    ax: Axes, x_vals: np.ndarray, label: Optional[str] = None, color: Optional = None
):
    ax.plot(x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], color, label=label)
    if label:
        ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
    else:
        ax.set(xticks=[], yticks=[], zticks=[])


def plot_test_trajectories(
    x_test: np.ndarray, x_sim: np.ndarray, t_test: np.ndarray, t_sim: np.ndarray
) -> None:
    """Plot a test trajectory

    Args:
        last_test: a single trajectory of the system
        model: a trained model to simulate and compare to test data
        dt: the time interval in test data

    Returns:
        A dict with two keys, "t_sim" (the simulation times) and
    "x_sim" (the simulated trajectory)
    """
    _, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    plt.suptitle("Test Trajectories by Dimension")
    plot_test_sim_data_1d_panel(axs, x_test, x_sim, t_test, t_sim)
    axs[-1].legend()

    plt.suptitle("Full Test Trajectories")
    if x_test.shape[1] == 2:
        _, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        _plot_test_sim_data_2d(axs, x_test, x_sim)
    elif x_test.shape[1] == 3:
        _, axs = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={"projection": "3d"})
        _plot_test_sim_data_3d(axs[0], x_test, "True Trajectory", "k")
        _plot_test_sim_data_3d(axs[1], x_sim, "Simulation", "r--")

    else:
        raise ValueError("Can only plot 2d or 3d data.")
    axs[0].set(title="true trajectory")
    axs[1].set(title="model simulation")
