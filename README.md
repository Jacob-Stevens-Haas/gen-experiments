# SINDy Experiments
`pysindy` experiments.  These experiments are meant to be run via [`mitosis`](https://github.com/Jacob-Stevens-Haas/mitosis),
but `mitosis` is not a requirement to run them.  It partially annotated with call signatures, and I'm happy to add more
as people point them out.

It's not yet on pypi, so install it with `pip install git+https://github.com/Jacob-Stevens-Haas/gen-experiments`
or clone and install it locally.

## Experiment Steps
There are three experiment steps made available:
* `gen_experiments.data.gen_data()`
* `gen_experiments.odes.run()`
* `gen_experiments.gridsearch.run()`

In addition, tentative PDE functionality is provided, though may be unstable:
* `gen_experiments.data.gen_pde_data()`
* `gen_experiments.pdes.run()`

## Plotting and diagnostics
Perhaps of most significance are the SINDy diagnostic plotting.  `gen_experiments.plotting` has a variety
of functions for creating diagnostics of fitted SINDy models:
* `gen_experiments.plotting.compare_coefficient_plots()` and its cousin,
  `gen_experiments.utils.unionize_coeff_matrices()`, which is used to align coefficient matrices from
  models with different features.
* `gen_experiments.utils.coeff_metrics()` and `gen_experiments.utils.pred_metrics()`]
* `gen_experiments.plotting.plot_training_data()`
* `gen_experiments.plotting.plot_test_trajectories()`


## Names
This package is distributed as pysindy-experiments, while some names still refer to gen_experiments.
The latter is due to the origins in a PhD general exam.

# Vendored branch
This branch removes the `pysindy` and `derivative` dependencies
    so that other projects which tie together tight development
            of `pysindy` and `pysindy-experiments`
        can manage these packages' compatible versions.
Do not develop off of this branch!
Instead, add all features to `main`, then rebase this branch at the tip.
