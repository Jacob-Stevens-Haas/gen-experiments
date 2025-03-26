from collections.abc import Sequence
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Callable, Optional, cast
from warnings import warn

import mitosis
import numpy as np
import scipy

from .gridsearch.typing import GridsearchResultDetails
from .odes import ode_setup
from .pdes import pde_setup
from .plotting import plot_training_data
from .typing import Float1D, Float2D, ProbData

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
TRIALS_FOLDER = Path(__file__).parent.absolute() / "trials"
MOD_LOG = getLogger(__name__)


def gen_data(
    group: str,
    seed: Optional[int] = None,
    n_trajectories: int = 1,
    ic_stdev: float = 3,
    noise_abs: Optional[float] = None,
    noise_rel: Optional[float] = None,
    dt: float = 0.01,
    t_end: float = 10,
    display: bool = False,
) -> dict[str, Any]:
    """Generate random training and test data

    An Experiment step according to the mitosis experiment runner.
    Note that test data has no noise.

    Arguments:
        group: the function to integrate
        seed (int): the random seed for number generation
        n_trajectories (int): number of trajectories of training data
        ic_stdev (float): standard deviation for generating initial conditions
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise-to-signal power ratio.
            Either noise_abs or noise_rel must be None.  Defaults to
            None.
        dt: time step for sample
        t_end: end time of simulation
        display: Whether to display graphics of generated data.

    Returns:
        dictionary of data and descriptive information
    """
    coeff_true = ode_setup[group]["coeff_true"]
    input_features = ode_setup[group]["input_features"]
    rhsfunc = ode_setup[group]["rhsfunc"]
    try:
        x0_center = ode_setup[group]["x0_center"]
    except KeyError:
        x0_center = np.zeros((len(input_features)), dtype=np.float_)
    try:
        nonnegative = ode_setup[group]["nonnegative"]
    except KeyError:
        nonnegative = False
    if noise_abs is not None and noise_rel is not None:
        raise ValueError("Cannot specify both noise_abs and noise_rel")
    elif noise_abs is None and noise_rel is None:
        noise_abs = 0.1

    MOD_LOG.info(f"Generating {n_trajectories} trajectories of f{group}")
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = _gen_data(
        rhsfunc,
        len(input_features),
        seed,
        x0_center=x0_center,
        nonnegative=nonnegative,
        n_trajectories=n_trajectories,
        ic_stdev=ic_stdev,
        noise_abs=noise_abs,
        noise_rel=noise_rel,
        dt=dt,
        t_end=t_end,
    )
    if display:
        figs = plot_training_data(x_train[0], x_train_true[0])
        figs[0].suptitle("Sample Trajectory")
    return {
        "data": ProbData(
            dt,
            t_train,
            x_train,
            x_test,
            x_dot_test,
            x_train_true,
            input_features,
            coeff_true,
        ),
        "main": f"{n_trajectories} trajectories of {rhsfunc}",
        "metrics": {"rel_noise": noise_rel, "abs_noise": noise_abs},
    }


def _gen_data(
    rhs_func: Callable,
    n_coord: int,
    seed: Optional[int],
    n_trajectories: int,
    x0_center: Float1D,
    ic_stdev: float,
    noise_abs: Optional[float],
    noise_rel: Optional[float],
    nonnegative: bool,
    dt: float,
    t_end: float,
) -> tuple[float, Float1D, list[Float2D], list[Float2D], list[Float2D], list[Float2D]]:
    rng = np.random.default_rng(seed)
    t_train = np.arange(0, t_end, dt, dtype=np.float_)
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
    x_train = x_train + cast(float, noise_abs) * rng.standard_normal(x_train.shape)
    x_train = list(x_train)
    x_test = list(x_test)
    x_dot_test = list(x_dot_test)
    x_train_true = list(x_train_true)
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true


def gen_pde_data(
    group: str,
    init_cond: np.ndarray,
    seed: int | None = None,
    noise_abs: float | None = None,
    rel_noise: float | None = None,
) -> dict[str, Any]:
    """Generate PDE measurement data for training

    For simplicity, Trajectories have been removed,
    Test data is the same as Train data.

    Arguments:
        group: name of the PDE
        init_cond: Initial Conditions for the PDE
        seed (int): the random seed for number generation
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise relative to amplitude of
            true data.  Amplitude of data is calculated as the max value
             of the power spectrum.  Either noise_abs or noise_rel must
             be None.  Defaults to None.

    Returns:
        dt, t_train, x_train, x_test, x_dot_test, x_train_true
    """
    rhsfunc = pde_setup[group]["rhsfunc"]["func"]
    input_features = pde_setup[group]["input_features"]
    if rel_noise is None:
        rel_noise = 0.1
    spatial_grid = pde_setup[group]["spatial_grid"]
    spatial_args = [
        (spatial_grid[-1] - spatial_grid[0]) / len(spatial_grid),
        len(spatial_grid),
    ]
    time_args = pde_setup[group]["time_args"]
    dimension = pde_setup[group]["rhsfunc"]["dimension"]
    coeff_true = pde_setup[group]["coeff_true"]
    try:
        time_args = pde_setup[group]["time_args"]
    except KeyError:
        time_args = [0.01, 10]
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = _gen_pde_data(
        rhsfunc,
        init_cond,
        spatial_args,
        dimension,
        seed,
        noise_abs=noise_abs,
        noise_rel=rel_noise,
        dt=time_args[0],
        t_end=time_args[1],
    )
    return {
        "data": ProbData(
            dt,
            t_train,
            x_train,
            x_test,
            x_dot_test,
            x_train_true,
            input_features,
            coeff_true,
        ),
        "main": f"1 trajectories of {rhsfunc.__qualname__}",
        "metrics": {"rel_noise": rel_noise, "abs_noise": noise_abs},
    }


def _gen_pde_data(
    rhs_func: Callable,
    init_cond: np.ndarray,
    spatial_args: Sequence,
    dimension: int,
    seed: int | None,
    noise_abs: float | None,
    noise_rel: float | None,
    dt: float,
    t_end: int,
):
    if noise_abs is not None and noise_rel is not None:
        raise ValueError("Cannot specify both noise_abs and noise_rel")
    elif noise_abs is None and noise_rel is None:
        noise_abs = 0.1
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
            args=spatial_args,
            **INTEGRATOR_KEYWORDS,
        ).y.T
    )
    t, x = x_train[0].shape
    x_train = np.stack(x_train, axis=-1)
    if dimension == 1:
        pass
    elif dimension == 2:
        x_train = np.reshape(x_train, (t, int(np.sqrt(x)), int(np.sqrt(x)), 1))
    elif dimension == 3:
        x_train = np.reshape(
            x_train, (t, int(np.cbrt(x)), int(np.cbrt(x)), int(np.cbrt(x)), 1)
        )
    x_test = x_train
    x_test = np.moveaxis(x_test, -1, 0)
    x_dot_test = np.array(
        [
            [rhs_func(0, xij, spatial_args[0], spatial_args[1]) for xij in xi]
            for xi in x_test
        ]
    )
    if dimension == 1:
        x_dot_test = [np.moveaxis(x_dot_test, [0, 1], [-1, -2])]
        pass
    elif dimension == 2:
        x_dot_test = np.reshape(x_dot_test, (t, int(np.sqrt(x)), int(np.sqrt(x)), 1))
        x_dot_test = [np.moveaxis(x_dot_test, 0, -2)]
    elif dimension == 3:
        x_dot_test = np.reshape(
            x_dot_test, (t, int(np.cbrt(x)), int(np.cbrt(x)), int(np.cbrt(x)), 1)
        )
        x_dot_test = [np.moveaxis(x_dot_test, 0, -2)]
    x_train_true = np.copy(x_train)
    if noise_rel is not None:
        noise_abs = np.sqrt(_max_amplitude(x_test, axis=-2) * noise_rel)
    x_train = x_train + cast(float, noise_abs) * rng.standard_normal(x_train.shape)
    x_train = [np.moveaxis(x_train, 0, -2)]
    x_train_true = np.moveaxis(x_train_true, 0, -2)
    x_test = [np.moveaxis(x_test, [0, 1], [-1, -2])]
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true


def _max_amplitude(signal: np.ndarray, axis: int) -> float:
    return np.abs(scipy.fft.rfft(signal, axis=axis)[1:]).max() / np.sqrt(
        signal.shape[axis]
    )


def _signal_avg_power(signal: np.ndarray) -> float:
    return np.square(signal).mean()


def load_results(hexstr: str) -> GridsearchResultDetails:
    """Load the results that mitosis saves

    Args:
        hexstr: randomly-assigned identifier for the results to open
    """
    return mitosis.load_trial_data(hexstr, trials_folder=TRIALS_FOLDER)
