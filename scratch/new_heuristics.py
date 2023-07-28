from typing import Any, Callable, Tuple
from functools import lru_cache

import auto_ks as aks
import kalman
import numpy as np

from scipy import sparse as sp
from scipy import optimize

rng = np.random.default_rng(98)
ArrayLike = np.ndarray | sp.sparray

Trajectory = tuple[np.ndarray, ...]


def gen_trajectory(times: np.ndarray, proc_noise: float) -> Trajectory:
    dts = times[1:] - times[:-1]
    Qis = [
        proc_noise * np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]])
        for dt in dts
    ]
    d_traj = np.array([np.random.multivariate_normal(np.zeros(2), Qi) for Qi in Qis])
    traj = np.concatenate((np.zeros((1, 2)), np.cumsum(d_traj, axis=0)), axis=0)
    traj[1:, 1] += dts * traj[:-1, 0]
    return (traj[:, 1], traj[:, 0])


def restack(traj: Trajectory) -> np.ndarray:
    return np.vstack(traj[::-1]).ravel("F")


def kalman_matrices(
    times: np.ndarray, meas_ind: np.ndarray | None = None
) -> tuple[sp.sparray, sp.sparray, sp.sparray]:
    if meas_ind is None:
        meas_ind = times
    delta_times = times[1:] - times[:-1]
    n = len(times)
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for dt in delta_times
    ]
    spQinv = sp.block_diag([np.linalg.inv(Q) for Q in Qs])
    Qinv = (spQinv + spQinv.T) / 2  # force to be symmetric

    G_left = sp.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
    G_right = sp.eye(2 * (n - 1))
    align_cols = sp.csc_array((2 * (n - 1), 2))
    G = sp.hstack((G_left, align_cols)) + sp.hstack((align_cols, G_right))

    H = observation_operator(times, meas_ind)
    return Qinv, H, G


def observation_operator(times: np.ndarray, meas_ind: np.ndarray) -> sp.sparray:
    n = len(times)
    H = sp.lil_array((n, 2 * n))
    H[:, 1::2] = sp.eye(n)
    return H[meas_ind]


def solve_kalman(
    Qinv: sp.sparray,
    G: sp.sparray,
    H: sp.sparray,
    measurements: np.ndarray,
    alpha: float | None,
) -> Trajectory:
    n = len(measurements)
    rhs = H.T @ measurements.reshape((-1, 1))
    lhs = H.T @ H + G.T * alpha @ Qinv @ G
    sol = np.linalg.solve(lhs.toarray(), rhs)
    x_hat = (H @ sol).flatten()
    x_dot_hat = (H[:, list(range(1, 2 * n)) + [0]] @ sol).flatten()
    return x_hat, x_dot_hat


def find_alpha_complex_witheld(
    times: np.ndarray, measurements: np.ndarray, alpha0: float = 1, detail=False
):
    h = 1e-6
    witheld_mask = np.zeros_like(times, dtype=bool)
    witheld_mask[::4] = True
    train_measurements = measurements[~witheld_mask]
    validation_measurements = measurements[~witheld_mask]
    Qinv, H, G = kalman_matrices(times, ~witheld_mask)
    H_witheld = observation_operator(times, witheld_mask)
    rhs = H.T @ train_measurements.reshape((-1, 1))


@lru_cache(10)
def _loss_alpha(
    curr_alpha: complex,
    H_witheld: sp.sparray | sp.spmatrix,
    validation_measurements: np.ndarray,
    G: sp.sparray | sp.spmatrix,
    Qinv: sp.sparray | sp.spmatrix,
    H: sp.sparray | sp.spmatrix,
) -> tuple[float, float]:
    lhs = H.T @ H + G.T @ (curr_alpha * Qinv) @ G
    sol = np.linalg.solve(lhs.toarray(), rhs)
    loss = ((H_witheld @ sol - validation_measurements) ** 2).sum()
    return loss.real, loss.imag / h


def grad_alpha(curr_alpha):
    curr_alpha = curr_alpha + h * 1j
    return _loss_alpha(curr_alpha[0])[1]


def loss_alpha(curr_alpha):
    curr_alpha = curr_alpha + h * 1j
    return _loss_alpha(curr_alpha[0])[0]


def scalar_grad_check(x0: Any, dx: Any, loss_fun: Callable, grad_fun: Callable):
    f0 = loss_fun(x0)
    g0 = grad_fun(x0)
    f1 = loss_fun(x0 + dx)
    g1 = grad_fun(x0 + dx)
    # Should be O(dx^2)
    assert np.abs(g1 * dx + g0 * dx - 2 * (f1 - f0)) < dx * 10

    scalar_grad_check(np.array([alpha0]), 1e-6, loss_alpha, grad_alpha)

    res = optimize.minimize(
        loss_alpha, alpha0, jac=grad_alpha, method="l-bfgs-b", bounds=[(0, np.inf)]
    )
    if detail:
        return res
    return res.x


def find_alpha_generalized(
    times: np.ndarray, measurements: np.ndarray, alpha0: float = 1, detail=False
):
    """BROKEN: Find kalman parameter alpha using generalized method

    See Jonker et al, Efﬁcient Robust Parameter Identiﬁcation in
    Generalized Kalman Smoothing Models

    r = residual, e.g. Hx-z.  rp = process residual, rm = measurement residual
    f(a, y) =  objective function, eg. lp(rp)+lm(rm)
    H = f_y = G.T @ Q^{-1} G y + H.T @ R^{-1/2} (R^{-1/2} H y - z)
    H_y = G.T @ Q^{-1} @ G + H.T @ R^{-1} H
    y(a) = (1/a * G.T @ Q^{-1} + H.T @ R^{-1} H)^{-1} @ H.T @ R^{-1/2} @ z
    """
    z = measurements
    Qinv, H, G = kalman_matrices(times)
    # imlicitly, measurement variance = 1
    _, _, _, Rinv, _ = kalman.initialize_values(measurements, times, 1)

    def sol(a):
        x_hat, x_dot_hat, _, _ = kalman.solve(measurements, H, times, a, 1)
        return kalman.restack(x_hat, x_dot_hat)

    def v(a):
        x = sol(a[0])
        return (
            0.5 * ((x.T @ H.T - z.T) @ Rinv @ (H @ x - z))
            + 0.5 * a * x.T @ G.T @ Qinv @ G @ x
        )

    f_x = lambda x, a: H.T @ Rinv @ H @ x - H.T @ Rinv @ z + a * G.T @ Qinv @ G @ x
    f_a = lambda x, a: -1 / (2 * a**2) * x.T @ G.T @ Qinv @ G @ x
    f_ax = lambda x, a: -1 / a**2 * G.T @ Qinv @ G @ x
    f_aa = lambda x, a: 1 / a**3 * x.T @ G.T @ Qinv @ G @ x
    f_xx = lambda x, a: H.T @ Rinv @ H + 1 / a * G.T @ Qinv @ G
    v_a = lambda a: f_a(sol(a[0]), a[0])

    def v_aa(a):
        B = f_ax(sol(a[0]), a[0])
        return (
            f_aa(sol(a[0]), a[0])
            - B.T @ np.linalg.inv(f_xx(sol(a[0]), a[0]).toarray()) @ B
        )

    # kalman.gradient_test(v, v_a, alpha0)
    # kalman.gradient_test(v_a, v_aa, alpha0)
    # kalman.complex_step_test(v, v_a, a0)
    # kalman.complex_step_test(v_a, v_aa, a0)
    lbfgs_sol = optimize.minimize(v, alpha0, method="l-bfgs-b", jac=v_a)
    bfgs_sol = optimize.minimize(v, alpha0, method="bfgs", jac=v_a)
    newton_sol = optimize.minimize(v, alpha0, method="newton-cg", jac=v_a, hess=v_aa)
    if detail:
        return (lbfgs_sol, bfgs_sol, newton_sol)
    return newton_sol.x


def find_alpha_barratt(
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
    return est_alpha, info
