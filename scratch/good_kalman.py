# %%
import pysindy as ps
import gen_experiments
import numpy as np
import matplotlib.pyplot as plt
import warnings

# %%
seed = 34
group = "duff"
metrics = gen_experiments.metrics["lorenzk"]
sim_params = gen_experiments.sim_params["10x"]
diff_params = gen_experiments.diff_params["kalman"]
diff_params["alpha"] = 0.00005
feat_params = gen_experiments.feat_params["cubic"]
# opt_params = gen_experiments.opt_params["test"]
opt_params = {
    "optcls": "ensemble",
    "opt": ps.MIOSR(target_sparsity=4),
    "bagging": True,
    "n_models": 20,
}

# %% [markdown]

#  # Run with known-good alpha

# %%
gen_experiments.odes.run(seed, group, sim_params, diff_params, feat_params, opt_params)


# %%

from gen_experiments.odes import ode_setup, gen_data

rhsfunc = ode_setup[group]["rhsfunc"]
input_features = ode_setup[group]["input_features"]
coeff_true = ode_setup[group]["coeff_true"]
try:
    x0_center = ode_setup[group]["x0_center"]
except KeyError:
    x0_center = np.zeros(len(input_features))
try:
    nonnegative = ode_setup[group]["nonnegative"]
except KeyError:
    nonnegative = False
dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_data(
    rhsfunc,
    len(input_features),
    seed,
    x0_center=x0_center,
    nonnegative=nonnegative,
    **sim_params,
)

# %%
diffs = [trajectory[1:] - trajectory[:-1] for trajectory in x_train]
meas_variance = np.mean([trajectory.var(axis=0) for trajectory in diffs], axis=0)
dt = t_train[1] - t_train[0]
# %%


def heuristic_alpha(
    x_trajectories: list[np.ndarray],
    times: np.ndarray,
    meas_var: int | float = None,
    max_alpha: int | float = 1e3,
    max_trials: int = 10,
):
    """Calculate the heuristic value of the Kalman parameter

    Heuristic does sometimes give negative or very large values due to
    limited data or short timesteps.  If it fails, it tries again,
    thinning the data by taking larger timesteps.

    Only works on data with equidistant

    Args:
        x_trajectories: training data trajectories, each with time as
            the 0th axis and coordinate as the 1st axis
        times: the times of observation for all the trajectories.
        meas_var: measurement variance, if known.
        max_alpha: maximum alpha permissible.
        max_trials: number of attempts to find a useful heuristic.

    Returns:
        estimated coefficient of smoothness regularizer in kalman smoothing.
    """
    if meas_var is not None:
        max_trials = 10
        for attempt in range(max_trials):
            print(f"Attempt {attempt}")
            x_trajectories = [
                trajectory[:: attempt + 1] for trajectory in x_trajectories
            ]
            diffs = [trajectory[1:] - trajectory[:-1] for trajectory in x_trajectories]
            obs_variance = np.mean([(diff**2).mean(axis=0) for diff in diffs])
            dt = (times[1] - times[0]) * (attempt + 1)
            t_exponent = dt**3 / 3
            scaled_obs_variance = (obs_variance - 2 * meas_var) / t_exponent
            scaled_obs_variance = scaled_obs_variance
            result = meas_var / scaled_obs_variance
            if result > 0 and result < max_alpha:
                return result
        else:
            warnings.warn(
                "Unable to determine optimal parameters.  Perhaps your measurement"
                " variance estimate is too high?  If not, try collecting data for a"
                " longer interval with a larger timestep"
            )
    return 0.05  # most measurements are more exact than the process they are measuring


# %% [markdown]

# # Run with heuristic alpha

# %%
new_alpha = heuristic_alpha(x_train, t_train, 0.1)
print(new_alpha)
diff_params["alpha"] = new_alpha
gen_experiments.odes.run(seed, group, sim_params, diff_params, feat_params, opt_params)

# %% [markdown]

# # Run with default alpha

# %%
diff_params["alpha"] = heuristic_alpha(x_train, t_train)
gen_experiments.odes.run(seed, group, sim_params, diff_params, feat_params, opt_params)


# %%
#  # High noise
sim_params = gen_experiments.sim_params["hi-noise"]
