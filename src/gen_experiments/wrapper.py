from typing import Iterable, Callable

import numpy as np

import gen_experiments


def run(
    ex_name: str,
    seed: int,
    grid_params: Iterable,
    grid_values: Iterable,
    grid_decision: Iterable,
    other_params: dict,
    metrics: Iterable,
):
    """
    Run a grid-search wrapper of an experiment.

    Arguments:
        ex_name: an experiment registered in gen_experiments
        grid_params: kwarg names to grid and pass to
            experiment
        grid_values: kwarg values to grid.  Indices match grid_params
        grid_decision: What to do with each grid param, e.g.
            {"plot", "best"}.  Indices match grid_params.
        other_params: a dict of other kwargs to pass to experiment
        metrics: names of metrics to record from each wrapped experiment
    """
    base_ex, base_group = gen_experiments.experiments[ex_name]
    results_shape = (len(metrics), *(len(grid) for grid in grid_values))
    results = np.zeros(results_shape)
    gridpoint_selector = np.ndindex(results_shape)
    for ind in gridpoint_selector:
        new_seed = np.random.randint(1000)
        curr_vals = [val[ind] for val in grid_values]
        curr_kwargs = {key: val for key, val in zip(grid_params, curr_vals)}
        results[:, ind] = base_ex(
            new_seed, group=base_group, **curr_kwargs, **other_params, display=False
        )
    plot_param_inds = [ind for ind, val in enumerate(grid_decision) if val == "plot"]
    for param_ind in plot_param_inds:
        selection_results = np.moveaxis(results, param_ind + 1, 1)
        new_shape = selection_results.shape
        selection_results = selection_results.reshape(*new_shape[:2], -1).max(-1)


def _marginalize_grid_views(grid_decision: Iterable, results: np.ndarray):
    """Marginalize unnecessary dimensions by taking max across axes."""
    plot_param_inds = [ind for ind, val in enumerate(grid_decision) if val == "plot"]
    grid_searches = []
    for param_ind in plot_param_inds:
        selection_results = np.moveaxis(results, param_ind + 1, 1)
        new_shape = selection_results.shape
        selection_results = selection_results.reshape(*new_shape[:2], -1).max(-1)
        grid_searches.append(selection_results)
    return grid_searches

