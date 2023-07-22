#  %%
import kalman
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse, optimize, stats

from new_heuristics import find_alpha_complex_witheld, find_alpha_generalized

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
x_hat, x_dot_hat, _, _ = kalman.solve(measurements, H, times, meas_var, proc_var)


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
x_hat, x_dot_hat, _, _ = kalman.solve(measurements, H, times, meas_var, proc_var)


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


alphas = []
ress = []
for sim in range(n_sims):
    measurements, x, x_dot, H, times = kalman.gen_data(
        rng.integers(0,100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_complex_witheld(times, measurements, alpha0=1, detail=True)
    alphas.append(result.x[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas)
plt.savefig("alphadist.png")
pass

# %%
alphas = []
ress = []
for sim in range(n_sims):
    measurements, x, x_dot, H, times = kalman.gen_data(
        rng.integers(0,100), stop=stop, nt=nt, meas_var=meas_var, process_var=proc_var
    )
    result = find_alpha_generalized(times, measurements, alpha0=1, detail=True)
    alphas.append(result[-1].x[0])
    ress.append(result)


fig = plt.figure()
plt.hist(alphas)
plt.savefig("alphadist2.png")
pass

# %%
