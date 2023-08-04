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
nt = 1000

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

n_sims = 20
meas_var = 1
true_alpha = meas_var / proc_var
seeds = rng.integers(0, 1000, size=n_sims)

# %%
alphas1 = []
ress = []
for seed in seeds:
    measurements, x, x_dot, H, times = kalman.gen_data(
        seed, stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    measurements = np.reshape(measurements, (-1, 1))
    result = find_alpha_complex_witheld(times, measurements, alpha0=1e0, detail=True)
    alphas1.append(result.x[0])
    ress.append(result)


fig = plt.figure(figsize = [8, 4])
fig.suptitle(f"Performance of LBFGS/Complex Step GCV Search on {n_sims} simulations")
ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(alphas1, bins=np.geomspace(min(alphas1), max(alphas1), 2 * len(alphas1) // 3))
ax1.set_xscale("log")
ax1.set_title(r"Distribution of $\hat \alpha")
ax1.set_xlabel(r"$\hat \alpha$")
ax1.set_ylabel("Frequency")

ax2 = fig.add_subplot(1, 2, 2)
ax2.hist([res.nit for res in ress])
ax2.set_title("Distribution of LBFGS iterations")
ax2.set_xlabel("Iteration")
plt.tight_layout()
plt.savefig("complex_step_lbfgs2.png")

pass
# %%
nt = 1000
alphas = np.logspace(-7, 7, 29)
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
# seeds = [7, 57, 54, 86, 70]
seeds = rng.integers(10000, size=20)
alpha_mins = []
for seed in seeds:
    # seed = rng.integers(0, 100)
    print(seed)
    measurements, x, x_dot, H, times = kalman.gen_data(
        seed, stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    measurements = np.reshape(measurements, (-1, 1))
    valid_losses = []
    training_losses = []
    grads = []
    for alpha in alphas:
        witheld_mask = np.zeros_like(times, dtype=bool)
        witheld_mask[::4] = True
        H_witheld = observation_operator(times, witheld_mask)

        train_measurements = measurements[~witheld_mask]
        validation_measurements = measurements[witheld_mask]
        Qinv, H, G = kalman_matrices(times, ~witheld_mask)
        H_witheld = observation_operator(times, witheld_mask)
        rhs = H.T @ train_measurements.reshape((-1, 1))

        # loss_fun = lambda a: loss_validation_alpha(a, H_witheld, validation_measurements, G, Qinv, H, rhs, 1e-6)
        # loss_grad = lambda a: grad_validation_alpha(a, H_witheld, validation_measurements, G, Qinv, H, rhs, 1e-6)
        # scalar_grad_check(np.array([1]), 1e-6, loss_fun, loss_grad)


        v_loss = loss_validation_alpha(np.array([alpha]), H_witheld, validation_measurements,G, Qinv, H, rhs, 1e-6)
        # t_loss = loss_training_alpha(np.array([alpha]), train_measurements,G, Qinv, H, rhs, alpha * 1e-4)
        # grad = grad_validation_alpha(np.array([alpha]), H_witheld, validation_measurements,G, Qinv, H, rhs, 1e-6)
        valid_losses.append(v_loss)
        # training_losses.append(t_loss)
        # grads.append(grad)

    alpha_mins.append(alphas[::-1][np.argmin(valid_losses[::-1])])
    ax.semilogx(alphas, (valid_losses-min(valid_losses))/(max(valid_losses)-min(valid_losses)), color=f"C0")
    # ax3.semilogx(alphas, (training_losses-min(training_losses))/(max(training_losses)-min(training_losses)), color=f"C1")
    # ax2.semilogx(alphas, grads, color="C1")
    
ax.set_title("Validation loss with lotsa timepoints may have local minima?")
plt.savefig("comlex_loss.png")
fig = plt.figure()
plt.hist(alpha_mins, bins=np.geomspace(min(alphas), max(alphas), len(alphas)))
plt.xscale("log")
plt.savefig("alphadist2.png")



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
alphas3 = []
ress = []
for seed in seeds:
    measurements, x, x_dot, H, times = kalman.gen_data(
        seed, stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_barratt(times, measurements, alpha0=1e0, detail=True)
    alphas3.append(result[0])
    ress.append(result)


fig = plt.figure(figsize=[8, 4])
fig.suptitle(f"Performance of Boyd/Barrett GCV Search on {n_sims} simulations")
ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(alphas3, bins=np.geomspace(min(alphas3), max(alphas3), 2 * len(alphas3) // 3))
ax1.set_xscale("log")
ax1.set_title(r"Distribution of $\hat \alpha (\alpha_0=\alpha=1)$")
ax1.set_xlabel(r"$\hat \alpha$")
ax1.set_ylabel("Frequency")

ax2 = fig.add_subplot(1, 2, 2)
for res in ress:
    ax2.semilogy(res[1]["losses"], color="C0")
ax2.set_title("Loss during training to 200 iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Witheld loss")
fig.tight_layout()
plt.savefig("alphadist3.png")
# %%
