from typing import Iterable, Collection
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import gen_experiments

name = "gridsearch"


def run(
    seed: int,
    ex_name: str,
    grid_params: Collection,
    grid_vals: Collection,
    grid_decisions: Collection,
    other_params: dict,
    metrics: Collection = None,
):
    """Run a grid-search wrapper of an experiment.

    Arguments:
        ex_name: an experiment registered in gen_experiments
        grid_params: kwarg names to grid and pass to
            experiment
        grid_vals: kwarg values to grid.  Indices match grid_params
        grid_decisions: What to do with each grid param, e.g.
            {"plot", "best"}.  Indices match grid_params.
        other_params: a dict of other kwargs to pass to experiment
        metrics: names of metrics to record from each wrapped experiment
    """
    other_params = NestedDict(**other_params)
    base_ex, base_group = gen_experiments.experiments[ex_name]
    results_shape = (len(metrics), *(len(grid) for grid in grid_vals))
    results = np.zeros(results_shape)
    gridpoint_selector = np.ndindex(results_shape[1:])
    rng = np.random.default_rng(seed)
    for ind in gridpoint_selector:
        new_seed = rng.integers(1000)
        for axis_ind, key, val_list in zip(ind, grid_params, grid_vals):
            other_params[key] = val_list[axis_ind]
        curr_results = base_ex.run(new_seed, **other_params, display=False)
        results[(slice(None), *ind)] = [curr_results[metric] for metric in metrics]
    grid_searches = _marginalize_grid_views(grid_decisions, results)
    n_row = results_shape[0]
    n_col = len(grid_searches)
    fig, subplots = plt.subplots(
        n_row, n_col, sharey="row", sharex="col", squeeze=False
    )
    for m_ind_row, m_name in enumerate(metrics):
        for col, (param_name, x_ticks, param_search) in enumerate(
            zip(grid_params, grid_vals, grid_searches)
        ):
            ax = subplots[m_ind_row, col]
            ax.plot(x_ticks, param_search[m_ind_row], label=m_name)
            if m_ind_row == 0:
                ax.set_title(f"{param_name}")
            if col == 0:
                ax.set_ylabel(f"{m_name}")
    fig.suptitle(f"Grid Search on {base_ex.name}")
    fig.tight_layout()
    main_metric_ind = metrics.index("main") if "main" in metrics else 0
    return {
        "results": results,
        "main": max(grid[main_metric_ind].max() for grid in grid_searches),
    }


def _marginalize_grid_views(
    grid_decision: Iterable, results: np.ndarray
) -> list[np.ndarray]:
    """Marginalize unnecessary dimensions by taking max across axes."""
    plot_param_inds = [ind for ind, val in enumerate(grid_decision) if val == "plot"]
    grid_searches = []
    for param_ind in plot_param_inds:
        selection_results = np.moveaxis(results, param_ind + 1, 1)
        new_shape = selection_results.shape
        selection_results = selection_results.reshape(*new_shape[:2], -1).max(-1)
        grid_searches.append(selection_results)
    return grid_searches


class NestedDict(defaultdict):
    def __missing__(self, key):
        prefix, subkey = key.split(".", 1)
        return self[prefix][subkey]

    def __setitem__(self, key, value):
        if "." in key:
            prefix, suffix = key.split(".", 1)
            return self[prefix].__setitem__(suffix, value)
        else:
            return super().__setitem__(key, value)
