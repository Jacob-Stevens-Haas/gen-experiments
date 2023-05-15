from typing import Iterable, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np

import gen_experiments
from gen_experiments.utils import NestedDict, SeriesList, SeriesDef

name = "gridsearch"


def run(
    seed: int,
    ex_name: str,
    grid_params: Sequence[str],
    grid_vals: Sequence[Sequence],
    grid_decisions: Sequence[str],
    other_params: dict,
    series_params: Optional[SeriesList] = None,
    metrics: Optional[Sequence] = None,
    display: bool = True,
):
    """Run a grid-search wrapper of an experiment.

    Arguments:
        ex_name: an experiment registered in gen_experiments
        grid_params: kwarg names to grid and pass to
            experiment
        grid_vals: kwarg values to grid.  Indices match grid_params
        grid_decisions: What to do with each grid param, e.g.
            {"plot", "max"}.  Indices match grid_params.
        other_params: a dict of other kwargs to pass to experiment
        metrics: names of metrics to record from each wrapped experiment
        display: whether to plot results.
    """
    other_params = NestedDict(**other_params)
    base_ex, base_group = gen_experiments.experiments[ex_name]
    if series_params is None:
        series_params = SeriesList(None, None, [SeriesDef(ex_name, {}, [], [])])
        legends = False
    else:
        legends = True
    n_metrics = len(metrics)
    n_plotparams = len([decide for decide in grid_decisions if decide == "plot"])
    grid_searches = []
    if base_group is not None:
        other_params["group"] = base_group
    for series_data in series_params.series_list:
        if series_params.param_name is not None:
            other_params[series_params.param_name] = series_data.static_param
        new_grid_vals = grid_vals + series_data.grid_vals
        new_grid_params = grid_params + series_data.grid_params
        new_grid_decisions = grid_decisions + len(series_data.grid_params) * ["best"]
        full_results_shape = (len(metrics), *(len(grid) for grid in new_grid_vals))
        full_results = np.zeros(full_results_shape)
        gridpoint_selector = np.ndindex(full_results_shape[1:])
        rng = np.random.default_rng(seed)
        for ind in gridpoint_selector:
            new_seed = rng.integers(1000)
            for axis_ind, key, val_list in zip(ind, new_grid_params, new_grid_vals):
                other_params[key] = val_list[axis_ind]
            curr_results = base_ex.run(new_seed, **other_params, display=False)
            full_results[(slice(None), *ind)] = [
                curr_results[metric] for metric in metrics
            ]
        grid_searches.append(_marginalize_grid_views(new_grid_decisions, full_results))

    if display:
        fig, subplots = plt.subplots(
            n_metrics,
            n_plotparams,
            sharey="row",
            sharex="col",
            squeeze=False,
            figsize=(n_plotparams * 3, 0.5 + n_metrics * 2.25),
        )
        for series_data, series_name in zip(
            grid_searches, (ser.name for ser in series_params.series_list)
        ):
            plot(
                fig,
                subplots,
                metrics,
                grid_params,
                grid_vals,
                series_data,
                series_name,
                legends,
            )
        if series_params.print_name is not None:
            title = f"Grid Search on {series_params.print_name} in {ex_name}"
        else:
            title = f"Grid Search in {ex_name}"
        fig.suptitle(title)
        fig.tight_layout()

    main_metric_ind = metrics.index("main") if "main" in metrics else 0
    return {
        "results": grid_searches,
        "main": max(grid[main_metric_ind].max() for grid in grid_searches),
    }


def plot(fig, subplots, metrics, grid_params, grid_vals, grid_searches, name, legends):
    for m_ind_row, m_name in enumerate(metrics):
        for col, (param_name, x_ticks, param_search) in enumerate(
            zip(grid_params, grid_vals, grid_searches)
        ):
            ax = subplots[m_ind_row, col]
            ax.plot(x_ticks, param_search[m_ind_row], label=name)
            if m_ind_row == 0:
                ax.set_title(f"{param_name}")
            if col == 0:
                ax.set_ylabel(f"{m_name}")
    if legends:
        ax.legend()


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
