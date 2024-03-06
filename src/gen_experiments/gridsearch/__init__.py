from collections.abc import Iterable
from copy import copy
from functools import partial
from logging import getLogger
from pprint import pformat
from types import EllipsisType as ellipsis
from typing import Annotated, Any, Collection, Optional, Sequence, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import DTypeLike, NDArray
from scipy.stats import kstest

import gen_experiments
from gen_experiments import config
from gen_experiments.odes import plot_ode_panel
from gen_experiments.plotting import _PlotPrefs
from gen_experiments.typing import FloatND
from gen_experiments.utils import simulate_test_data

from .typing import (
    ExpResult,
    GridsearchResult,
    GridsearchResultDetails,
    NestedDict,
    OtherSliceDef,
    SavedGridPoint,
    SeriesDef,
    SeriesList,
    SkinnySpecs,
)

pformat = partial(pformat, indent=4, sort_dicts=True)
logger = getLogger(__name__)
name = "gridsearch"
lookup_dict = vars(config)


def _amax_to_full_inds(
    amax_inds: Collection[tuple[int | slice, int] | ellipsis] | ellipsis,
    amax_arrays: list[list[GridsearchResult[np.void]]],
) -> set[tuple[int, ...]]:
    """Find full indexers to selected elements of argmax arrays

    Args:
        amax_inds: selection statemtent of which argmaxes to return.
        amax_arrays: arrays of indexes to full gridsearch that are responsible for
            the computed max values.  First level of nesting reflects series, second
            level reflects which grid grid axis.
    Returns:
        all indexers to full gridsearch that are requested by amax_inds
    """

    def np_to_primitive(tuple_like: np.void) -> tuple[int, ...]:
        return tuple(int(el) for el in cast(Iterable, tuple_like))

    if amax_inds is ...:  # grab each element from arrays in list of lists of arrays
        return {
            np_to_primitive(el)
            for ar_list in amax_arrays
            for arr in ar_list
            for el in arr.flatten()
        }
    all_inds = set()
    for plot_axis_results in [el for series in amax_arrays for el in series]:
        for ind in amax_inds:
            if ind is ...:  # grab each element from arrays in list of lists of arrays
                all_inds |= {
                    np_to_primitive(el)
                    for ar_list in amax_arrays
                    for arr in ar_list
                    for el in arr.flatten()
                }
            elif isinstance(ind[0], int):
                all_inds |= {np_to_primitive(cast(np.void, plot_axis_results[ind]))}
            else:  # ind[0] is slice(None)
                all_inds |= {np_to_primitive(el) for el in plot_axis_results[ind]}
    return all_inds


_EqTester = TypeVar("_EqTester")


def _param_normalize(val: _EqTester) -> _EqTester | str:
    if type(val).__eq__ == object.__eq__:
        return repr(val)
    else:
        return val


def _grid_locator_match(
    exp_params: dict[str, Any],
    exp_ind: tuple[int, ...],
    param_spec: Collection[dict[str, Any]],
    ind_spec: Collection[tuple[int, ...]],
) -> bool:
    """Determine whether experimental parameters match a specification

    Logical clause applied is:

        OR((exp_params MATCHES params for params in param_spec))
        AND
        OR((exp_ind MATCHES ind for ind in ind_spec))

    Treats OR of an empty collection as falsy
    Args:
        exp_params: the experiment parameters to evaluate
        exp_ind: the experiemnt's full-size grid index to evaluate
        param_spec: the criteria for matching exp_params
        ind_spec: the criteria for matching exp_ind
    """
    found_match = False
    for params_or in param_spec:
        params_or = {k: _param_normalize(v) for k, v in params_or.items()}

        try:
            if all(
                _param_normalize(exp_params[param]) == value
                for param, value in params_or.items()
            ):
                found_match = True
                break
        except KeyError:
            pass
    for ind_or in ind_spec:
        # exp_ind doesn't include metric, so skip first metric
        if _index_in(exp_ind, ind_or[1:]):
            break
    else:
        return False
    return found_match


def run(
    seed: int,
    group: str,
    grid_params: list[str],
    grid_vals: list[Sequence],
    grid_decisions: list[str],
    other_params: dict,
    skinny_specs: SkinnySpecs,
    series_params: Optional[SeriesList] = None,
    metrics: Sequence[str] = (),
    plot_prefs: _PlotPrefs = _PlotPrefs(True, False, ()),
) -> GridsearchResultDetails:
    """Run a grid-search wrapper of an experiment.

    Arguments:
        group: an experiment registered in gen_experiments.  It must
            have a name and a metric_ordering attribute
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
            where axes are all searched, but not all combinations are
            searched.  The first element is a sequence of grid_names to
            skinnify.  The second is the thin_slices criteria (see
            docstring for _ndindex_skinny).  By default, all plot axes
            are made skinny with respect to each other.
    """
    other_params = NestedDict(**other_params)
    base_ex, base_group = gen_experiments.experiments[group]
    if series_params is None:
        series_params = SeriesList(None, None, [SeriesDef(group, {}, [], [])])
        legends = False
    else:
        legends = True
    n_metrics = len(metrics)
    metric_ordering = [base_ex.metric_ordering[metric] for metric in metrics]
    n_plotparams = len([decide for decide in grid_decisions if decide == "plot"])
    series_searches: list[tuple[list[GridsearchResult], list[GridsearchResult]]] = []
    intermediate_data: list[SavedGridPoint] = []
    plot_data: list[SavedGridPoint] = []
    if base_group is not None:
        other_params["group"] = base_group
    for s_counter, series_data in enumerate(series_params.series_list):
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
                ind for ind, decide in enumerate(new_grid_decisions) if decide == "plot"
            ]
            where_others = None
        full_results_shape = (len(metrics), *(len(grid) for grid in new_grid_vals))
        full_results = np.full(full_results_shape, np.nan)
        gridpoint_selector = _ndindex_skinny(
            full_results_shape[1:], ind_skinny, where_others
        )
        rng = np.random.default_rng(seed)
        for ind_counter, ind in enumerate(gridpoint_selector):
            print(f"Calculating series {s_counter}, gridpoint{ind_counter}", end="\r")
            new_seed = rng.integers(1000)
            for axis_ind, key, val_list in zip(ind, new_grid_params, new_grid_vals):
                curr_other_params[key] = val_list[axis_ind]
            curr_results, grid_data = base_ex.run(
                new_seed, **curr_other_params, display=False, return_all=True
            )
            grid_data: ExpResult
            intermediate_data.append(
                {"params": curr_other_params.flatten(), "pind": ind, "data": grid_data}
            )
            full_results[(slice(None), *ind)] = [
                curr_results[metric] for metric in metrics
            ]
        grid_optima, grid_ind = _marginalize_grid_views(
            new_grid_decisions, full_results, metric_ordering
        )
        series_searches.append((grid_optima, grid_ind))

    if plot_prefs:
        full_m_inds = _amax_to_full_inds(
            plot_prefs.grid_ind_match, [s[1] for s in series_searches]
        )
        for int_data in intermediate_data:
            logger.debug(
                f"Checking whether to save/plot :\n{pformat(int_data['params'])}\n"
                f"\tat location {pformat(int_data['pind'])}\n"
                f"\tagainst spec: {pformat(plot_prefs.grid_params_match)}\n"
                f"\twith allowed locations {pformat(full_m_inds)}"
            )
            if _grid_locator_match(
                int_data["params"],
                int_data["pind"],
                plot_prefs.grid_params_match,
                full_m_inds,
            ) and int_data["params"] not in [saved["params"] for saved in plot_data]:
                grid_data = int_data["data"]
                print("Results for params: ", int_data["params"], flush=True)
                grid_data |= simulate_test_data(
                    grid_data["model"], grid_data["dt"], grid_data["x_test"]
                )
                logger.info("Found match, simulating and plotting")
                plot_ode_panel(grid_data)
                plot_data.append(int_data)
        if plot_prefs.rel_noise:
            grid_vals, grid_params = plot_prefs.rel_noise(
                grid_vals, grid_params, grid_data
            )
        fig, subplots = plt.subplots(
            n_metrics,
            n_plotparams,
            sharey="row",
            sharex="col",
            squeeze=False,
            figsize=(n_plotparams * 3, 0.5 + n_metrics * 2.25),
        )
        for series_data, series_name in zip(
            series_searches, (ser.name for ser in series_params.series_list)
        ):
            plot(
                subplots,
                metrics,
                grid_params,
                grid_vals,
                series_data[0],
                series_name,
                legends,
            )
        if series_params.print_name is not None:
            title = f"Grid Search on {series_params.print_name} in {group}"
        else:
            title = f"Grid Search in {group}"
        fig.suptitle(title)
        fig.tight_layout()

    main_metric_ind = metrics.index("main") if "main" in metrics else 0
    return {
        "system": group,
        "plot_data": plot_data,
        "series_data": {
            name: data
            for data, name in zip(
                [list(zip(metrics, argopts)) for metrics, argopts in series_searches],
                [ser.name for ser in series_params.series_list],
            )
        },
        "metrics": metrics,
        "grid_params": grid_params,
        "grid_vals": grid_vals,
        "main": max(
            grid[main_metric_ind].max()
            for metrics, _ in series_searches
            for grid in metrics
        ),
    }


def plot(
    subplots: NDArray[Annotated[np.void, "Axes"]],
    metrics: Sequence[str],
    grid_params: Sequence[str],
    grid_vals: Sequence[Sequence[float] | np.ndarray],
    grid_searches: Sequence[GridsearchResult],
    name: str,
    legends: bool,
):
    if len(metrics) == 0:
        raise ValueError("Nothing to plot")
    for m_ind_row, m_name in enumerate(metrics):
        for col, (param_name, x_ticks, param_search) in enumerate(
            zip(grid_params, grid_vals, grid_searches)
        ):
            ax = cast(Axes, subplots[m_ind_row, col])
            ax.plot(x_ticks, param_search[m_ind_row], label=name)
            x_ticks = np.array(x_ticks)
            if m_name in ("coeff_mse", "coeff_mae"):
                ax.set_yscale("log")
            x_ticks_normalized = (x_ticks - x_ticks.min()) / (
                x_ticks.max() - x_ticks.min()
            )
            x_ticks_lognormalized = (np.log(x_ticks) - np.log(x_ticks).min()) / (
                np.log(x_ticks.max()) - np.log(x_ticks).min()
            )
            ax = subplots[m_ind_row, col]
            if kstest(x_ticks_normalized, "uniform") < kstest(
                x_ticks_lognormalized, "uniform"
            ):
                ax.set_xscale("log")
            if m_ind_row == 0:
                ax.set_title(f"{param_name}")
            if col == 0:
                ax.set_ylabel(f"{m_name}")
    if legends:
        ax.legend()  # type: ignore


T = TypeVar("T", bound=np.generic)


def _argopt(
    arr: FloatND, axis: Optional[int | tuple[int, ...]] = None, opt: str = "max"
) -> NDArray[np.void]:
    """Calculate the argmax/min, but accept tuple axis.

    Ignores NaN values

    Args:
        arr: an array to search
        axis: The axis or axes to search through for the argmax/argmin.
        opt: One of {"max", "min"}

    Returns:
        array of indices for the argopt.  If m = arr.ndim and n = len(axis),
        the final result will be an array of ndim = m-n with elements being
        tuples of length m
    """
    dtype: DTypeLike = [(f"f{axind}", "i") for axind in range(arr.ndim)]
    if axis is None:
        axis = ()
    axis = (axis,) if isinstance(axis, int) else axis
    keep_axes = tuple(sorted(set(range(arr.ndim)) - set(axis)))
    keep_shape = tuple(arr.shape[ax] for ax in keep_axes)
    result = np.empty(keep_shape, dtype=dtype)
    optfun = np.nanargmax if opt == "max" else np.nanargmin
    for slise in np.ndindex(keep_shape):
        sub_arr = arr
        # since we shrink shape, we need to chop of axes from the end
        for ind, ax in zip(reversed(slise), reversed(keep_axes)):
            sub_arr = np.take(sub_arr, ind, ax)
        subind_max = np.unravel_index(optfun(sub_arr), sub_arr.shape)
        fullind_max = np.empty((arr.ndim), int)
        fullind_max[np.array(keep_axes, int)] = slise
        fullind_max[np.array(axis, int)] = subind_max
        result[slise] = tuple(fullind_max)
    return result


def _marginalize_grid_views(
    grid_decisions: Iterable[str],
    results: Annotated[NDArray[T], "shape (n_metrics, *n_gridsearch_values)"],
    max_or_min: Sequence[str],
) -> tuple[list[GridsearchResult[T]], list[GridsearchResult]]:
    """Marginalize unnecessary dimensions by taking max across axes.

    Ignores NaN values
    Args:
        grid_decisions: list of how to treat each non-metric gridsearch
            axis.  An array of metrics for each "plot" grid decision
            will be returned, along with an array of the the index
            of collapsed dimensions that returns that metric
        results: An array of shape (n_metrics, *n_gridsearch_values)
        max_or_min: either "max" or "min" for each row of results
    Returns:
        a list of the metric optima for each plottable grid decision, and
        a list of the flattened argoptima.
    """
    arg_dtype = np.dtype(",".join(results.ndim * "i"))
    plot_param_inds = [ind for ind, val in enumerate(grid_decisions) if val == "plot"]
    grid_searches = []
    args_maxes = []
    optfuns = [np.nanmax if opt == "max" else np.nanmin for opt in max_or_min]
    for param_ind in plot_param_inds:
        reduce_axes = tuple(set(range(results.ndim - 1)) - {param_ind})
        selection_results = np.array(
            [opt(result, axis=reduce_axes) for opt, result in zip(optfuns, results)]
        )
        sub_arrs = []
        for m_ind, (result, opt) in enumerate(zip(results, max_or_min)):

            def _metric_pad(tp: tuple[int, ...]) -> np.void:
                return np.void((m_ind, *tp), dtype=arg_dtype)

            pad_m_ind = np.vectorize(_metric_pad)
            arg_max = pad_m_ind(_argopt(result, reduce_axes, opt))
            sub_arrs.append(arg_max)

        args_max = np.stack(sub_arrs)
        grid_searches.append(selection_results)
        args_maxes.append(args_max)
    return grid_searches, args_maxes


def _ndindex_skinny(
    shape: tuple[int, ...],
    thin_axes: Optional[Sequence[int]] = None,
    thin_slices: Optional[Sequence[OtherSliceDef]] = None,
):
    """
    Return an iterator like ndindex, but only traverse thin_axes once

    This is useful for grid searches with multiple plot axes, where
    searching across all combinations of plot axes is undesirable.
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
        thin_axes = ()
        thin_slices = ()
    elif thin_axes is None:
        raise ValueError("Must pass thin_axes if thin_slices is not None")
    elif thin_slices is None:  # slice other thin axes at 0th index
        n_thin = len(thin_axes)
        thin_slices = n_thin * ((n_thin - 1) * (0,),)
    full_indexes = np.ndindex(shape)
    thin_slices = cast(Sequence[OtherSliceDef], thin_slices)

    def ind_checker(multi_index: tuple[int, ...]) -> bool:
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
                match &= multi_index[ax2] == range(shape[ax2])[slice_ind]
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
        grid_params.index(pname) for pname in skinny_specs[0] if pname in grid_params
    ]
    missing_sk_inds = [
        skinny_specs[0].index(pname)
        for pname in skinny_specs[0]
        if pname not in grid_params
    ]
    where_others = []
    for orig_sk_ind, match_criteria in zip(
        range(len(skinny_specs[0])), skinny_specs[1], strict=True
    ):
        if orig_sk_ind in missing_sk_inds:
            continue
        missing_criterion_inds = tuple(
            sk_ind if sk_ind < orig_sk_ind else sk_ind - 1 for sk_ind in missing_sk_inds
        )
        new_criteria = tuple(
            match_criterion
            for cr_ind, match_criterion in enumerate(match_criteria)
            if cr_ind not in missing_criterion_inds
        )
        where_others.append(new_criteria)
    return skinny_param_inds, tuple(where_others)


def strict_find_grid_match(
    results: GridsearchResultDetails,
    *,
    params: Optional[dict[str, Any]] = None,
    ind_spec: Optional[tuple[int | slice, int] | ellipsis] = None,
) -> ExpResult:
    if params is None:
        params = {}
    if ind_spec is None:
        ind_spec = ...
    matches = []
    amax_arrays = [
        [single_ser_and_axis[1] for single_ser_and_axis in single_series_all_axes]
        for _, single_series_all_axes in results["series_data"].items()
    ]
    full_inds = _amax_to_full_inds((ind_spec,), amax_arrays)

    for trajectory in results["plot_data"]:
        if _grid_locator_match(
            trajectory["params"], trajectory["pind"], (params,), full_inds
        ):
            matches.append(trajectory)

    if len(matches) > 1:
        raise ValueError("Specification is nonunique; matched multiple results")
    if len(matches) == 0:
        raise ValueError("Could not find a match")
    return matches[0]["data"]


def _index_in(base: tuple[int, ...], tgt: tuple[int | ellipsis | slice, ...]) -> bool:
    """Determine whether base indexing tuple will match given numpy index"""
    if len(base) > len(tgt):
        return False
    curr_ax = 0
    for ax, ind in enumerate(tgt):
        if isinstance(ind, int):
            try:
                if ind != base[curr_ax]:
                    return False
            except IndexError:
                return False
        elif isinstance(ind, slice):
            if not (ind.start is None and ind.stop is None and ind.step is None):
                raise ValueError("Only slices allowed are `slice(None)`")
        elif ind is ...:
            base_ind_remaining = len(base) - curr_ax
            tgt_ind_remaining = len(tgt) - ax
            # ellipsis can take 0 or more spots
            curr_ax += max(base_ind_remaining - tgt_ind_remaining, -1)
        curr_ax += 1
    if curr_ax == len(base):
        return True
    return False
