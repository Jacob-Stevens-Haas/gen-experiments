#  %%
import auto_ks as aks
import kalman
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse, optimize, stats

from new_heuristics import (
    observation_operator,
    kalman_matrices,
    find_alpha_complex_witheld,
    find_alpha_generalized,
    find_alpha_barratt,
    loss_validation_alpha,
    loss_training_alpha,
    grad_validation_alpha,
    scalar_grad_check
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
ax = fig.add_subplot(1, 1, 1)
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
ax = fig.add_subplot(1, 1, 1)
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
meas_var = 0.5
true_alpha = meas_var / proc_var

# %%
alphas1 = []
ress = []
for sim in range(n_sims):
    measurements, x, x_dot, H, times = kalman.gen_data(
        rng.integers(0, 100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_complex_witheld(times, measurements, alpha0=1, detail=True)
    alphas1.append(result.x[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas1)
plt.savefig("alphadist.png")
pass
# %%
alphas = np.logspace(-8, 7, 21)
fig = plt.figure()
ax = fig.gca()
# ax2 = plt.twinx(ax)
ax3 = plt.twinx(ax)
ax.set_xlabel("alpha")
ax.set_ylabel("validation loss", color="C0")
ax.tick_params("y", color="C0", labelcolor="C0")
# ax2.set_ylim(-1, 1)
# ax2.set_ylabel("gradient", color="C1")
# ax2.tick_params("y", color="C1", labelcolor="C1")
ax3.set_ylabel("training loss", color="C1")
ax3.tick_params("y", color="C1", labelcolor="C1")
seeds = [7, 57, 54, 86, 70]
for trial in range(5):
    # seed = rng.integers(0, 100)
    seed = seeds[trial]
    print(seed)
    measurements, x, x_dot, H, times = kalman.gen_data(
        seed, stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    valid_losses = []
    training_losses = []
    grads = []
    for alpha in alphas:
        witheld_mask = np.zeros_like(times, dtype=bool)
        witheld_mask[::4] = True
        H_witheld = observation_operator(times, witheld_mask)

        train_measurements = measurements[~witheld_mask]
        validation_measurements = measurements[~witheld_mask]
        Qinv, H, G = kalman_matrices(times, ~witheld_mask)
        H_witheld = observation_operator(times, witheld_mask)
        rhs = H.T @ train_measurements.reshape((-1, 1))

        loss_fun = lambda a: loss_validation_alpha(a, H_witheld, validation_measurements, G, Qinv, H, rhs, 1e-6)
        loss_grad = lambda a: grad_validation_alpha(a, H_witheld, validation_measurements, G, Qinv, H, rhs, 1e-6)
        scalar_grad_check(np.array([1]), 1e-6, loss_fun, loss_grad)


        v_loss = loss_validation_alpha(np.array([alpha]), H_witheld, validation_measurements,G, Qinv, H, rhs, 1e-6)
        t_loss = loss_training_alpha(np.array([alpha]), train_measurements,G, Qinv, H, rhs, 1e-6)
        grad = grad_validation_alpha(np.array([alpha]), H_witheld, validation_measurements,G, Qinv, H, rhs, 1e-6)
        valid_losses.append(v_loss)
        training_losses.append(t_loss)
        grads.append(grad)

    ax.semilogx(alphas, (valid_losses-min(valid_losses))/(max(valid_losses)-min(valid_losses)), color=f"C0")
    ax3.semilogx(alphas, (training_losses-min(training_losses))/(max(training_losses)-min(training_losses)), color=f"C1")
    # ax2.semilogx(alphas, grads, color="C1")
    
ax.set_title("Training Loss is pointed the wrong way\nValidation loss may have local minima")
plt.savefig("comlex_loss.png")


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
        rng.integers(0, 100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_barratt(times, measurements, alpha0=1, detail=True)
    alphas3.append(result[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas3)
plt.savefig("alphadist3.png")

pass
# %%
