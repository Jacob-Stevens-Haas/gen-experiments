#  %%
import auto_ks as aks
import kalman
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse, optimize, stats

from new_heuristics import (
    find_alpha_complex_witheld,
    find_alpha_generalized,
    find_alpha_barratt
)

# %%
seed = 98
rng = np.random.default_rng(seed)
stop = 2 * np.pi
nt = 100

times = np.linspace(0, stop, nt)
meas_var = 1
proc_var = 1

# %%
measurements, x, x_dot, H, times = kalman.gen_sine(seed, stop=stop, nt=nt, meas_var=1)
state_vec = kalman.restack(x, x_dot)
x_hat, x_dot_hat, _, _, _ = kalman.solve(measurements, H, times, meas_var, proc_var)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ln1 = ax.plot(times, x, label="x", color="C0")
ln2 = ax.plot(times, x_hat, ".", label=r"$\hat x$", color="C0")
lnz = ax.plot(times, measurements, "x", color="C2", label="z")

ax2 = ax.twinx()
ln3 = ax2.plot(times, x_dot, label=r"\dot x", color="C1")
ln4 = ax2.plot(times, x_dot_hat, ".", label=r"$\hat {\dot x}$", color="C1")
lns = ln1 + ln2 + lnz + ln3 + ln4
ax.set_title("Sinusoidal Example")
plt.legend(lns, [ln.get_label() for ln in lns])


# %%
measurements, x, x_dot, H, times = kalman.gen_data(
    seed, stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
)
state_vec = kalman.restack(x, x_dot)
x_hat, x_dot_hat, _, _, _ = kalman.solve(measurements, H, times, meas_var, proc_var)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ln1 = ax.plot(times, x, label="x", color="C0")
ln2 = ax.plot(times, x_hat, ".", label=r"$\hat x$", color="C0")
lnz = ax.plot(times, measurements, "x", color="C2", label="z")

ax2 = ax.twinx()
ln3 = ax2.plot(times, x_dot, label=r"\dot x", color="C1")
ln4 = ax2.plot(times, x_dot_hat, ".", label=r"$\hat {\dot x}$", color="C1")
lns = ln1 + ln2 + lnz + ln3 + ln4
ax.set_title("True Kalman Process Example")
plt.legend(lns, [ln.get_label() for ln in lns])

plt.savefig("generated.png")

# %%

n_sims = 10
meas_var = .5
true_alpha = meas_var / proc_var

# %%
alphas1 = []
ress = []
for sim in range(n_sims):
    measurements, x, x_dot, H, times = kalman.gen_data(
        rng.integers(0,100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_complex_witheld(times, measurements, alpha0=1, detail=True)
    alphas1.append(result.x[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas1)
plt.savefig("alphadist.png")
pass

# %%
# alphas2 = []
# ress = []
# for sim in range(n_sims):
#     measurements, x, x_dot, H, times = kalman.gen_data(
#         rng.integers(0,100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
#     )
#     result = find_alpha_generalized(times, measurements, alpha0=1, detail=True)
#     alphas2.append(result[-1].x[0])
#     ress.append(result)


# fig = plt.figure()
# plt.hist(alphas2)
# plt.savefig("alphadist2.png")

# %%
# def find_alpha_barratt(
#     times: np.ndarray,
#     measurements: np.ndarray,
#     alpha0: float = 1,
#     detail=False
# ):
#     """Find kalman parameter alpha using GCV error

#     See Boyd & Barratt, Fitting a Kalman Smoother to Data.  No regularization
#     """
#     measurements = measurements.reshape((-1, 1))
#     nt = len(measurements)
#     dt = times[1] - times[0]
#     Ai = np.array([[1, 0], [dt, 1]])
#     Qi = kalman.gen_Qi(dt)
#     Qi_rt_inv = np.linalg.cholesky(np.linalg.inv(Qi))
#     Qi_r_i_vec = np.reshape(Qi_rt_inv, (-1,1))
#     Qi_proj = lambda vec: Qi_r_i_vec @ (Qi_r_i_vec.T @ Qi_r_i_vec) ** -1 @ (Qi_r_i_vec.T) @ vec
#     Hi = np.array([[0, 1]])
#     Ri = np.eye(1)
#     Ri_rt_inv = Ri
#     params0 = aks.KalmanSmootherParameters(Ai, Qi_rt_inv, Hi, Ri)
#     mask = np.ones_like(measurements, dtype=bool)
#     mask[::4] = False
#     def proj(curr_params, t):
#         W_n_s_v = np.reshape(curr_params.W_neg_sqrt, (-1,1))
#         W_n_s_v = np.reshape(Qi_proj(W_n_s_v), (2,2))
#         new_params = aks.KalmanSmootherParameters(
#             Ai, W_n_s_v, Hi, Ri_rt_inv
#         )
#         return new_params, t
#     params, info = aks.tune(params0, proj, measurements, K=mask, lam=.1, verbose=False)
#     est_Q = np.linalg.inv(params.W_neg_sqrt @ params.W_neg_sqrt.T)
#     est_alpha = 1 / (est_Q / Qi).mean()
#     return est_alpha, info

# %%
alphas3 = []
ress = []
for sim in range(n_sims):
    measurements, x, x_dot, H, times = kalman.gen_data(
        rng.integers(0,100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_barratt(times, measurements, alpha0=1, detail=True)
    alphas3.append(result[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas3)
plt.savefig("alphadist3.png")

pass
# %%
