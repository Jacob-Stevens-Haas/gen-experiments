from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from matplotlib.axes import Axes

from .gridsearch.typing import GridLocator

PAL = sns.color_palette("Set1")
PLOT_KWS = {"alpha": 0.7, "linewidth": 3}


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
):
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

        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax


def compare_coefficient_plots(
    coefficients_est: Annotated[np.ndarray, "(n_coord, n_feat)"],
    coefficients_true: Annotated[np.ndarray, "(n_coord, n_feat)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
):
    """Create plots of true and estimated coefficients."""
    n_cols = len(coefficients_est)

    # helps boost the color of small coefficients.  Maybe log is better?
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


def plot_training_trajectory(
    ax: Axes,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: np.ndarray,
    labels: bool = True,
) -> None:
    """Plot a single training trajectory"""
    if x_train.shape[1] == 2:
        ax.plot(x_true[:, 0], x_true[:, 1], ".", label="True", color=PAL[0], **PLOT_KWS)
        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            ".",
            label="Measured",
            color=PAL[1],
            **PLOT_KWS,
        )
        if np.linalg.norm(x_smooth - x_train) / x_smooth.size > 1e-12:
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                ".",
                label="Smoothed",
                color=PAL[2],
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
        if np.linalg.norm(x_smooth - x_train) / x_smooth.size > 1e-12:
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                x_smooth[:, 2],
                ".",
                color=PAL[2],
                label="Smoothed values",
                alpha=0.3,
            )
        if labels:
            ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        else:
            ax.set(xticks=[], yticks=[], zticks=[])
    else:
        raise ValueError("Can only plot 2d or 3d data.")


def plot_training_data(x_train: np.ndarray, x_true: np.ndarray, x_smooth: np.ndarray):
    """Plot training data (and smoothed training data, if different)."""
    fig = plt.figure(figsize=(12, 6))
    if x_train.shape[-1] == 2:
        ax0 = fig.add_subplot(1, 2, 1)
    elif x_train.shape[-1] == 3:
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    else:
        raise ValueError("Too many or too few coordinates to plot")
    plot_training_trajectory(ax0, x_train, x_true, x_smooth)
    ax0.legend()
    ax0.set(title="Training data")
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.loglog(np.abs(scipy.fft.rfft(x_train, axis=0)) / np.sqrt(len(x_train)))
    ax1.set(title="Training Data Absolute Spectral Density")
    ax1.set(xlabel="Wavenumber")
    ax1.set(ylabel="Magnitude")
    return fig


def plot_pde_training_data(x_train, x_true, x_smooth, rel_noise):
    """Plot and compare true data, training data and smoothed data for PDEs."""
    # 1D:
    if x_train.shape[-1] == 1:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axs[0].imshow(x_true, vmin=0, vmax=x_true.max())
        axs[0].set(title="True Data")
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(x_train, vmin=0, vmax=x_true.max())
        axs[1].set(title=f"Noisy Data with {rel_noise} Relative Noise")
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(x_smooth, vmin=0, vmax=x_smooth.max())
        smoothing_error = np.sqrt(((x_true - x_smooth) ** 2).mean())
        axs[2].set(title=f"Smoothed Data with smoothing error {smoothing_error}")
        fig.colorbar(im2, ax=axs[2])
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 6))
        fft_values = np.abs(scipy.fft.rfft(x_train - x_true, axis=-2)) / np.sqrt(
            x_train.shape[-2]
        )
        plt.loglog(fft_values.mean(axis=0))
        plt.title("Measurement Noise Spectrum")
        plt.xlabel("Wavenumber")
        plt.ylabel("Magnitude")
        plt.show()


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
    axs: Annotated[Sequence[Axes], "len=3"],
    x_test: np.ndarray,
    x_sim: np.ndarray,
    labels: bool = True,
) -> None:
    axs[0].plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], "k", label="True Trajectory")
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], "r--", label="Simulation")
    for ax in axs:
        if labels:
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
        _plot_test_sim_data_3d(axs, x_test, x_sim)
    else:
        raise ValueError("Can only plot 2d or 3d data.")
    axs[0].set(title="true trajectory")
    axs[1].set(title="model simulation")
