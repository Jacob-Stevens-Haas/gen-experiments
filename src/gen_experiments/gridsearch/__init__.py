from collections.abc import Collection, Iterable
from copy import copy
from functools import partial
from logging import getLogger
from pprint import pformat
from time import process_time
from types import EllipsisType as ellipsis
from typing import Annotated, Any, Callable, Optional, Sequence, TypeVar, Union, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import DTypeLike, NDArray
from scipy.stats import kstest

import gen_experiments

from .. import config
from ..data import gen_data, gen_pde_data
from ..odes import plot_ode_panel
from ..pdes import plot_pde_panel
from ..plotting import _PlotPrefs
from ..typing import FloatND, NestedDict
from ..utils import simulate_test_data
from .typing import (
    ExpResult,
    GridLocator,
    GridsearchResult,
    GridsearchResultDetails,
    KeepAxisSpec,
    OtherSliceDef,
    SavedGridPoint,
    SeriesData,
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

    if amax_inds is ...:  # grab each element from arrays in list of lists of arrays
        return {
            void_to_tuple(el)
            for ar_list in amax_arrays
            for arr in ar_list
            for el in arr.flatten()
        }
    all_inds = set()
    for plot_axis_results in [el for series in amax_arrays for el in series]:
        for ind in amax_inds:
            if ind is ...:  # grab each element from arrays in list of lists of arrays
                all_inds |= {
                    void_to_tuple(el)
                    for ar_list in amax_arrays
                    for arr in ar_list
                    for el in arr.flatten()
                }
            elif isinstance(ind[0], int):
                all_inds |= {void_to_tuple(cast(np.void, plot_axis_results[ind]))}
            else:  # ind[0] is slice(None)
                all_inds |= {void_to_tuple(el) for el in plot_axis_results[ind]}
    return all_inds


_EqTester = TypeVar("_EqTester")


def _param_normalize(val: _EqTester) -> _EqTester | str:
    """Allow equality testing of mutable objects with useful reprs"""
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
    warn("Use find_gridpoints() instead", DeprecationWarning)
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
    skinny_specs: Optional[SkinnySpecs] = None,
    series_params: Optional[SeriesList] = None,
    metrics: tuple[str, ...] = (),
    plot_prefs: _PlotPrefs = _PlotPrefs(),
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
    logger.info(f"Beginning gridsearch of system: {group}")
    other_params = NestedDict(**other_params)
    base_ex, base_group = gen_experiments.experiments[group]
    if base_ex.__name__ == "gen_experiments.odes":
        plot_panel = plot_ode_panel
        data_step = gen_data
    elif base_ex.__name__ == "gen_experiments.pdes":
        plot_panel = plot_pde_panel
        data_step = gen_pde_data
    elif base_ex.__name__ == "NoExperiment":
        data_step = gen_experiments.NoExperiment.gen_data
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
        for ind_counter, ind in enumerate(gridpoint_selector):
            logger.info(
                f"Calculating series {s_counter} ({series_data.name}), "
                f"gridpoint {ind_counter} ({ind})"
            )
            start = process_time()
            for axis_ind, key, val_list in zip(ind, new_grid_params, new_grid_vals):
                curr_other_params[key] = val_list[axis_ind]
            sim_params = curr_other_params.pop("sim_params", {})
            group_arg = curr_other_params.pop("group", None)
            data = data_step(seed=seed, group=group_arg, **sim_params)["data"]
            curr_results, grid_data = base_ex.run(
                data, **curr_other_params, display=False, return_all=True
            )
            curr_results["sim_params"] = sim_params
            curr_results["group"] = group
            intermediate_data.append(
                {"params": curr_other_params.flatten(), "pind": ind, "data": grid_data}
            )
            full_results[(slice(None), *ind)] = [
                curr_results[metric] for metric in metrics
            ]
            logger.info(f"Last calculation: {process_time() - start:.2f} sec.")
        grid_optima, grid_ind = _marginalize_grid_views(
            new_grid_decisions, full_results, metric_ordering
        )
        series_searches.append((grid_optima, grid_ind))

    main_metric_ind = metrics.index("main") if "main" in metrics else 0
    scan_grid = {
        p: v for p, d, v in zip(grid_params, grid_decisions, grid_vals) if d == "plot"
    }
    results: GridsearchResultDetails = {
        "system": group,
        "plot_data": [],
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
        "scan_grid": scan_grid,
        "plot_grid": {},
        "main": max(
            grid[main_metric_ind].max()
            for metrics, _ in series_searches
            for grid in metrics
        ),
    }
    if plot_prefs:
        plot_data = []
        # todo - improve how plot_prefs.plot_match interacts with series
        # This is a horrible hack, assuming a params_or for each series, ino
        if len(series_params.series_list) != len(plot_prefs.plot_match.params_or):
            msg = (
                "Trying to plot a subset of points tends to require the same"
                "number of matchable parameter lists as series, lined up 1:1."
                "You have a different number of each."
            )
            # TODO: write a warn_external function in mitosis for this:
            warn(msg)
            logger.warning(msg)
        for series_data, params in zip(
            series_params.series_list,
            list(plot_prefs.plot_match.params_or),
        ):
            key = series_data.name
            logger.info(f"Searching for matching points in series: {key}")
            start = process_time()
            locator = GridLocator(
                plot_prefs.plot_match.metrics, plot_prefs.plot_match.keep_axes, [params]
            )
            plot_data += find_gridpoints(
                locator,
                intermediate_data,
                [results["series_data"][key]],
                results["metrics"],
                results["scan_grid"],
            )
            logger.info(f"Searching took {process_time() - start:.2f} sec")
        results["plot_data"] = plot_data
        for gridpoint in plot_data:
            grid_data = gridpoint["data"]
            logger.info(f"Plotting: {gridpoint['params']}")
            start = process_time()
            grid_data |= simulate_test_data(
                grid_data["model"], grid_data["dt"], grid_data["x_test"]
            )
            plot_panel(grid_data)  # type: ignore
            logger.info(f"Sim/Plot took {process_time() - start:.2f} sec")
        if plot_prefs.rel_noise:
            raise ValueError("_PlotPrefs.rel_noise is not correctly implemented.")
        else:
            results["plot_grid"] = scan_grid

        if n_plotparams > 0:
            fig, subplots = plt.subplots(
                n_metrics,
                n_plotparams,
                sharey="row",
                sharex="col",
                squeeze=False,
                figsize=(n_plotparams * 3, 0.5 + n_metrics * 2.25),
            )
            for series_search, series_name in zip(
                series_searches, (ser.name for ser in series_params.series_list)
            ):
                plot(
                    subplots,
                    metrics,
                    cast(Sequence[str], results["plot_grid"].keys()),
                    cast(Sequence[Sequence], results["plot_grid"].values()),
                    series_search[0],
                    series_name,
                    legends,
                )
            if series_params.print_name is not None:
                title = f"Grid Search on {series_params.print_name} in {group}"
            else:
                title = f"Grid Search in {group}"
            fig.suptitle(title)
            fig.tight_layout()

    return results


def plot(
    subplots: NDArray[Annotated[np.void, "Axes"]],
    metrics: Sequence[str],
    plot_params: Sequence[str],
    grid_vals: Sequence[Sequence[float] | np.ndarray],
    grid_searches: Sequence[GridsearchResult],
    name: str,
    legends: bool,
):
    if len(metrics) == 0:
        raise ValueError("Nothing to plot")
    for m_ind_row, m_name in enumerate(metrics):
        for col, (param_name, x_ticks, param_search) in enumerate(
            zip(plot_params, grid_vals, grid_searches)
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
) -> tuple[list[GridsearchResult[T]], list[GridsearchResult[np.void]]]:
    """Marginalize unnecessary dimensions by taking max or min across axes.

    Ignores NaN values and strips the metric index from the argoptima.

    Args:
        grid_decisions: list of how to treat each non-metric gridsearch
            axis.  An array of metrics for each "plot" grid decision
            will be returned, along with an array of the the index
            of collapsed dimensions that returns that metric
        results: An array of shape (n_metrics, *n_gridsearch_values)
        max_or_min: either "max" or "min" for each row of results
    Returns:
        a list of the metric optima for each plottable grid decision, and
        a list of the flattened argoptima, with metric removed
    """
    plot_param_inds = [ind for ind, val in enumerate(grid_decisions) if val == "plot"]
    if not plot_param_inds:
        plot_param_inds = [results.ndim - 1]
    grid_searches = []
    args_maxes = []
    optfuns = [np.nanmax if opt == "max" else np.nanmin for opt in max_or_min]
    for param_ind in plot_param_inds:
        reduce_axes = tuple(set(range(results.ndim - 1)) - {param_ind})
        selection_results = np.array(
            [opt(result, axis=reduce_axes) for opt, result in zip(optfuns, results)]
        )
        sub_arrs = []
        for result, opt in zip(results, max_or_min):
            arg_max = _argopt(result, reduce_axes, opt)
            sub_arrs.append(arg_max)

        args_max = np.stack(sub_arrs)
        grid_searches.append(selection_results.reshape((len(max_or_min), -1)))
        args_maxes.append(args_max.reshape((len(max_or_min), -1)))
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
            indexes.  Empty sequence or None (default) is to thin no axes.
        thin_slices: the indexes for other thin axes when traversing
            a particular thin axis. Defaults to 0th index

    Example:

    >>> set(_ndindex_skinny((2,2), (0,1), ((0,), (lambda x: x,))))

    {(0, 0), (0, 1), (1, 1)}
    """
    full_indexes = np.ndindex(shape)
    if not thin_axes and thin_slices is None:
        yield from full_indexes
    elif thin_axes is None:
        raise ValueError("Must pass thin_axes if thin_slices is not None")
    elif thin_slices is None:  # slice other thin axes at 0th index
        n_thin = len(thin_axes)
        thin_slices = n_thin * ((n_thin - 1) * (0,),)
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


def find_gridpoints(
    find: GridLocator,
    where: list[SavedGridPoint],
    argopt_arrs: Collection[SeriesData],
    argopt_metrics: Sequence[str],
    argopt_axes: dict[str, Sequence[object]],
) -> list[SavedGridPoint]:
    """Find results wrapped by gridsearch that match criteria

    Args:
        find: the criteria
        where: The list of saved gridpoints to search
        context: The overall data for the gridsearch, describing metrics, grid
            setup, and gridsearch results

    Returns:
        A list of the matching points in the gridsearch.
    """
    results: list[SavedGridPoint] = []
    partial_match: set[tuple[int, ...]] = set()
    inds_of_metrics: Sequence[int]
    if find.metrics is ...:
        inds_of_metrics = range(len(argopt_metrics))
    else:
        inds_of_metrics = tuple(argopt_metrics.index(metric) for metric in find.metrics)
    # No deduplication is done!
    keep_axes = _normalize_keep_axes(find.keep_axes, argopt_axes)

    ser: list[tuple[GridsearchResult[np.floating], GridsearchResult[np.void]]]
    for ser in argopt_arrs:
        for index_of_ax, indexes_in_ax in keep_axes:
            amax_arr = ser[index_of_ax][1]
            amax_want = amax_arr[np.ix_(inds_of_metrics, indexes_in_ax)].flatten()
            partial_match |= {void_to_tuple(el) for el in amax_want}
    logger.info(
        f"Found {len(partial_match)} gridpoints that match metric-plot_axis criteria"
    )

    params_or = tuple(
        {k: v if callable(v) else _param_normalize(v) for k, v in params_match.items()}
        for params_match in find.params_or
    )

    def check_values(criteria: Any | Callable[..., bool], candidate: Any) -> bool:
        if callable(criteria):
            return criteria(candidate)
        else:
            return _param_normalize(candidate) == criteria

    for point in filter(lambda p: p["pind"] in partial_match, where):
        logger.debug(f"Checking whether {point['pind']} matches param query")
        for params_match in params_or:
            if all(
                param in point["params"] and check_values(value, point["params"][param])
                for param, value in params_match.items()
            ):
                results.append(point)
                break

    logger.info(f"found {len(results)} points that match all GridLocator criteria")
    return results


def _normalize_keep_axes(
    keep_axes: KeepAxisSpec, scan_grid: dict[str, Sequence[Any]]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    ax_sizes = {ax_name: len(vals) for ax_name, vals in scan_grid.items()}
    if ... in keep_axes:
        keep_axes = _expand_ellipsis_axis(keep_axes, ax_sizes)  # type: ignore
    else:
        keep_axes = cast(Collection[tuple[str, tuple[int, ...]]], keep_axes)
    scan_axes = tuple(ax_sizes.keys())
    return tuple((scan_axes.index(keep_ax[0]), keep_ax[1]) for keep_ax in keep_axes)


def _expand_ellipsis_axis(
    keep_axis: Union[
        tuple[ellipsis, ellipsis],
        tuple[ellipsis, tuple[int, ...]],
        tuple[tuple[str, ...], ellipsis],
    ],
    ax_sizes: dict[str, int],
) -> Collection[tuple[str, tuple[int, ...]]]:
    if keep_axis[0] is ... and keep_axis[1] is ...:
        # form 1
        return tuple((k, tuple(range(v))) for k, v in ax_sizes.items())
    elif isinstance(keep_axis[1], tuple):
        # form 2
        return tuple((k, keep_axis[1]) for k in ax_sizes.keys())
    elif isinstance(keep_axis[0], tuple):
        # form 3
        return tuple((k, tuple(range(ax_sizes[k]))) for k in keep_axis[0])
    else:
        raise TypeError("Keep_axis does not have an ellipsis or is not a 2-tuple")


def void_to_tuple(tuple_like: np.void) -> tuple[int, ...]:
    """Turn a void that represents a tuple of ints into a tuple of ints"""
    return tuple(int(el) for el in cast(Iterable, tuple_like))
