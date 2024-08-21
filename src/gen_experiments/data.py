from math import ceil
from pathlib import Path
from typing import Callable, Optional, cast
from warnings import warn

import mitosis
import numpy as np
import scipy

from gen_experiments.gridsearch.typing import GridsearchResultDetails
from gen_experiments.utils import Float1D, Float2D

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
TRIALS_FOLDER = Path(__file__).parent.absolute() / "trials"


def gen_data(
    rhs_func: Callable,
    n_coord: int,
    seed: Optional[int] = None,
    n_trajectories: int = 1,
    x0_center: Optional[Float1D] = None,
    ic_stdev: float = 3,
    noise_abs: Optional[float] = None,
    noise_rel: Optional[float] = None,
    nonnegative: bool = False,
    dt: float = 0.01,
    t_end: float = 10,
) -> tuple[float, Float1D, list[Float2D], list[Float2D], list[Float2D], list[Float2D]]:
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
        x0_center = np.zeros((n_coord), dtype=np.float_)
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
        dimension: Number of spatial dimensions (1, 2, or 3)
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
            args=args,
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
        [[rhs_func(0, xij, args[0], args[1]) for xij in xi] for xi in x_test]
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
