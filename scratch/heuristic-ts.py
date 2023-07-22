# %%

import numpy as np
import matplotlib.pyplot as plt
from derivative import Kalman
import warnings

warnings.resetwarnings()

# %% [markdown]

# We want to generate a sample trajectory with a known estimated process 
# variance = 1

# %%

dt = 1
T = 1000
t = np.arange(0,T,dt)


rng = np.random.default_rng(1)
meas_var = .01
dx = rng.normal(scale = np.sqrt(dt **3 /3), size = t.shape)
x = dx.cumsum()

# %% [markdown]

# Now, we run some trials to determine the distribution of estimated process
# variances

# %%
n_trials = 100
est_proc_vars = []
for _ in range(n_trials):
    err = np.sqrt(meas_var) * rng.standard_normal(size = t.shape)
    z = x + err
    alpha = Kalman._heuristic_alpha(z, t, meas_var, max_trials=1)
    est_proc_vars.append(meas_var / alpha)

# %%

plt.hist(est_proc_vars, density=True)
plt.title("With a good amount of widely-spaced data,\n we can get a pretty good distribution")
plt.xlabel(f"Estimated Proc variance (correct = {meas_var})")
plt.ylabel("Density")
# %% [markdown]
# Heuristic changes with dt
# We want to know how the estimated variance changes with dt for constant
# number of timesteps

# %%
rng = np.random.default_rng(1)

N = 1000
n_trials = 20
fig = plt.figure()
ax = plt.gca()
ax.set_title(f"Distribution by dt for {N} steps")
all_data = []
dts = np.logspace(0, -1, 4)
for dt in dts:
    T = N * dt
    t = np.arange(0, T, dt)
    meas_var = .01
    dx = rng.normal(scale = np.sqrt(dt **3 /3), size = t.shape)
    x = dx.cumsum()
    est_proc_vars = []
    for _ in range(n_trials):
        err = np.sqrt(meas_var) * rng.standard_normal(size = t.shape)
        z = x + err
        alpha = Kalman._heuristic_alpha(z, t, meas_var, max_trials=1)
        est_proc_vars.append(alpha)
    all_data.append(est_proc_vars)

ax.set_xscale("log")
bins = np.logspace(
    np.log10(np.array(all_data).min()),
    np.log10(np.array(all_data).max()),
    100
)
for trial_ind, dt in enumerate(dts):
    ax.hist(all_data[trial_ind], bins=bins, label=f"{dt:.03}", log=True, density=True)
ax.legend()
ax.set_xlabel(f"Estimated alpha (variance ratio).  Correct is {meas_var}, default=.05")
# %% [markdown]
# and by number of timesteps

# %%
rng = np.random.default_rng(1)

dt = .1
n_trials = 20
fig = plt.figure()
ax = plt.gca()
ax.set_title(f"Distribution by N for dt={dt} (correct is {meas_var})")
all_data = []
for N in np.logspace(1, 4, 4):
    T = N * dt
    t = np.arange(0, T, dt)
    meas_var = .01
    dx = rng.normal(scale = np.sqrt(dt **3 /3), size = t.shape)
    x = dx.cumsum()
    est_proc_vars = []
    for _ in range(n_trials):
        err = np.sqrt(meas_var) * rng.standard_normal(size = t.shape)
        z = x + err
        alpha = Kalman._heuristic_alpha(z, t, meas_var, max_trials=1)
        est_proc_vars.append(alpha)
    all_data.append(est_proc_vars)

ax.set_xscale("log")
bins = np.logspace(
    np.log10(np.array(all_data).min()),
    np.log10(np.array(all_data).max()),
    100
)
for trial_ind, N in enumerate(np.logspace(1, 4, 4)):
    ax.hist(all_data[trial_ind], bins=bins, label=N, log=True, density=True)
ax.legend()
ax.set_xlabel(f"Estimated alpha (variance ratio).  Correct is {meas_var}, default=.05")

# %% [markdown]

# Increasing dt makes the distribution much tighter around 1
# %%
rng = np.random.default_rng(1)

dt = 1
n_trials = 20
fig = plt.figure()
ax = plt.gca()
ax.set_title(f"Distribution by N for dt={dt} (correct is {meas_var})")
all_data = []
for N in np.logspace(1, 4, 4):
    T = N * dt
    t = np.arange(0, T, dt)
    meas_var = .01
    dx = rng.normal(scale = np.sqrt(dt **3 /3), size = t.shape)
    x = dx.cumsum()
    est_proc_vars = []
    for _ in range(n_trials):
        err = np.sqrt(meas_var) * rng.standard_normal(size = t.shape)
        z = x + err
        alpha = Kalman._heuristic_alpha(z, t, meas_var, max_trials=1)
        est_proc_vars.append(alpha)
    all_data.append(est_proc_vars)

ax.set_xscale("log")
bins = np.logspace(
    np.log10(np.array(all_data).min()),
    np.log10(np.array(all_data).max()),
    100
)
for trial_ind, N in enumerate(np.logspace(1, 4, 4)):
    ax.hist(all_data[trial_ind], bins=bins, label=N, log=True, density=True)
ax.legend()
ax.set_xlabel(f"Estimated alpha (variance ratio).  Correct is {meas_var}, default=.05")
