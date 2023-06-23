from copy import copy
from typing import Iterable, Sequence, Optional, Callable, Collection

from scipy.stats import kstest
import matplotlib.pyplot as plt
import numpy as np

import gen_experiments
from gen_experiments.utils import (
    _PlotPrefs,
    NestedDict,
    SeriesList,
    SeriesDef,
    plot_training_data,
    plot_test_trajectories,
    compare_coefficient_plots,
    _max_amplitude
)
name = "gridsearch"
OtherSliceDef = tuple[int | Callable]
SkinnySpecs = Optional[tuple[tuple[str, ...], tuple[OtherSliceDef, ...]]]

def run(
    seed: int,
    ex_name: str,
    grid_params: list[str],
    grid_vals: list[Sequence],
    grid_decisions: Sequence[str],
    other_params: dict,
    series_params: Optional[SeriesList] = None,
    metrics: Optional[Sequence] = None,
    plot_prefs: _PlotPrefs = _PlotPrefs(True, False, ()),
    skinny_specs: SkinnySpecs = None,
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
        plot_prefs: whether to plot results, and if so, a function to
            intercept and modify plot data.  Use this for applying any
            scaling or conversions.
        skinny_specs: Allow only conducting some of the grid search,
            where axes are all searched, but not all combinates are
            searched.  The first element is a sequence of grid_names to
            skinnify.  The second is the thin_slices criteria (see
            docstring for _ndindex_skinny).  By default, all plot axes
            are made skinny with respect to each other.
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
        curr_other_params = copy(other_params)
        if series_params.param_name is not None:
            curr_other_params[series_params.param_name] = series_data.static_param
        new_grid_vals: list = grid_vals + series_data.grid_vals
        new_grid_params = grid_params + series_data.grid_params
        new_grid_decisions = grid_decisions + len(series_data.grid_params) * ["best"]
        if skinny_specs is not None:
            ind_skinny, where_others = _curr_skinny_specs(skinny_specs, new_grid_params)
        else:
            ind_skinny = [
                ind for ind, decide in enumerate(new_grid_decisions) if decide=="plot"
            ]
            where_others = None
        full_results_shape = (len(metrics), *(len(grid) for grid in new_grid_vals))
        full_results = np.empty(full_results_shape)
        full_results.fill(-np.inf)
        gridpoint_selector = _ndindex_skinny(full_results_shape[1:], ind_skinny, where_others)
        rng = np.random.default_rng(seed)
        for ind in gridpoint_selector:
            new_seed = rng.integers(1000)
            for axis_ind, key, val_list in zip(ind, new_grid_params, new_grid_vals):
                curr_other_params[key] = val_list[axis_ind]
            curr_results, recent_data = base_ex.run(
                new_seed, **curr_other_params, display=False, return_all=True
            )
            if _params_match(curr_other_params, plot_prefs.grid_plot_match) and plot_prefs:
                plot_gridpoint(recent_data, curr_other_params)
            full_results[(slice(None), *ind)] = [
                curr_results[metric] for metric in metrics
            ]
        grid_searches.append(_marginalize_grid_views(new_grid_decisions, full_results))

    if plot_prefs:
        if plot_prefs.rel_noise:
            grid_vals, grid_params = plot_prefs.rel_noise(grid_vals, grid_params, recent_data)
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
            x_ticks = np.array(x_ticks)
            if m_name in ("coeff_mse", "coeff_mae"):
                ax.set_yscale("log")
            x_ticks_normalized = (
                (x_ticks - x_ticks.min())
                / (x_ticks.max() - x_ticks.min())
            )
            x_ticks_lognormalized = (
                (np.log(x_ticks) - np.log(x_ticks).min())
                / (np.log(x_ticks.max()) - np.log(x_ticks).min())
            )
            ax = subplots[m_ind_row, col]
            if (
                kstest(x_ticks_normalized, "uniform")
                < kstest(x_ticks_lognormalized, "uniform")
            ):
                ax.set_xscale("log")
            if m_ind_row == 0:
                ax.set_title(f"{param_name}")
            if col == 0:
                ax.set_ylabel(f"{m_name}")
    if legends:
        ax.legend()


def _params_match(exp_params: dict, plot_params: Collection[dict]) -> bool:
    """Determine whether experimental parameters match a specification"""
    for pref_or in plot_params:
        try:
            if all(exp_params[param] == value for param, value in pref_or.items()):
                return True
        except KeyError:
            pass
    return False


def plot_gridpoint(grid_data: dict, other_params: dict):
    print("Results for params: ", other_params, flush=True)
    sim_ind = -1
    x_train = grid_data["x_train"][sim_ind]
    x_true = grid_data["x_train_true"][sim_ind]
    model = grid_data["model"]
    model.print()
    smooth_train = model.differentiation_method.smoothed_x_
    plot_training_data(x_train, x_true, smooth_train)
    compare_coefficient_plots(
        grid_data["coefficients"],
        grid_data["coeff_true"],
        input_features=grid_data["input_features"],
        feature_names=grid_data["feature_names"],
    )
    plot_test_trajectories(grid_data["x_test"][sim_ind], model, grid_data["dt"])
    plt.show()


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

def _ndindex_skinny(
        shape: tuple[int],
        thin_axes: Optional[Sequence[int]] = None,
        thin_slices: Optional[Sequence[OtherSliceDef]] = None
    ):
    """
    Return an iterator like ndindex, but only traverse thin_axes once

    This is useful for grid searches with multiple plot axes, where
    searching across all combinations of plot axes is undesireable.
    Slow for big arrays! (But still probably trivial compared to the
    gridsearch operation :))

    Args:
        shape: array shape
        thin_axes: axes for which you don't want the product of all
            indexes
        thin_slices: the indexes for other thin axes when traversing
            a particular thin axis. Defaults to 0th index

    Example:

    >>> set(_ndindex_skinny((2,2), (0,1), ((0,), (lambda x: x,))))

    {(0, 0), (0, 1), (1, 1)}
    """
    if thin_axes is None and thin_slices is None:
        thin_axes = tuple()
        thin_slices = tuple()
    elif thin_axes is None:
        raise ValueError("Must pass thin_axes if thin_slices is not None")
    elif thin_slices is None:  # slice other thin axes at 0th index
        n_thin = len(thin_axes)
        thin_slices = (n_thin * ((n_thin-1) * (0,),))
    full_indexes = np.ndindex(shape)

    def ind_checker(multi_index):
        """Check if a multi_index meets thin index criteria"""
        matches = []
        # check whether multi_index matches criteria of any thin_axis
        for ax1, where_others in zip(thin_axes, thin_slices, strict=True):
            other_axes = list(thin_axes)
            other_axes.remove(ax1)
            match = True
            # check whether multi_index meets criteria of a particular thin_axis
            for ax2, slice_ind in zip(other_axes, where_others, strict=True):
                if callable(slice_ind):
                    slice_ind = slice_ind(multi_index[ax1])
                # would check: "== slice_ind", but must allow slice_ind = -1
                match *= (multi_index[ax2] == range(shape[ax2])[slice_ind])
            matches.append(match)
        return any(matches)

    while True:
        try:
            ind = next(full_indexes)
        except StopIteration:
            break
        if ind_checker(ind):
            yield ind


def _curr_skinny_specs(
    skinny_specs: SkinnySpecs, grid_params: list[str]
) -> tuple[Sequence[int], Sequence[OtherSliceDef]]:
    """Calculate which skinny specs apply to current parameters"""
    skinny_param_inds = [
        grid_params.index(pname)
        for pname in skinny_specs[0]
        if pname in grid_params
    ]
    missing_sk_inds = [
        skinny_specs[0].index(pname)
        for pname in skinny_specs[0]
        if pname not in grid_params
    ]
    where_others = []
    for orig_sk_ind, match_criteria in zip(
        range(len(skinny_specs[0])),
        skinny_specs[1],
        strict=True
    ):
        if orig_sk_ind in missing_sk_inds:
            continue
        missing_criterion_inds = tuple(
            sk_ind if sk_ind < orig_sk_ind else sk_ind-1 for sk_ind in missing_sk_inds
        )
        new_criteria = tuple(
            match_criterion
            for cr_ind, match_criterion in enumerate(match_criteria)
            if cr_ind not in missing_criterion_inds
        )
        where_others.append(new_criteria)
    return skinny_param_inds, tuple(where_others)
