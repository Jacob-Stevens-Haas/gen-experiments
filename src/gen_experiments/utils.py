from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from types import EllipsisType as ellipsis
from typing import (
    Annotated,
    Any,
    Collection,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)
from warnings import warn

import auto_ks as aks
import kalman
import numpy as np
import pysindy as ps
import sklearn
import sklearn.metrics
from numpy.typing import DTypeLike, NBitBase, NDArray

NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Shape = TypeVar("Shape", bound=tuple[int, ...])
FloatND = np.ndarray[Shape, np.dtype[np.floating[NBitBase]]]


class SINDyTrialData(TypedDict):
    dt: float
    coeff_true: Annotated[Float2D, "(n_coord, n_features)"]
    coeff_fit: Annotated[Float2D, "(n_coord, n_features)"]
    feature_names: Annotated[list[str], "length=n_features"]
    input_features: Annotated[list[str], "length=n_coord"]
    t_train: Float1D
    x_train: np.ndarray
    x_true: np.ndarray
    smooth_train: np.ndarray
    x_test: np.ndarray
    x_dot_test: np.ndarray
    model: ps.SINDy


class SINDyTrialUpdate(TypedDict):
    t_sim: Float1D
    t_test: Float1D
    x_sim: FloatND


class FullSINDyTrialData(SINDyTrialData):
    t_sim: Float1D
    x_sim: np.ndarray


class SavedData(TypedDict):
    params: dict
    pind: tuple[int]
    data: SINDyTrialData | FullSINDyTrialData


T = TypeVar("T", bound=np.generic)
GridsearchResult = Annotated[NDArray[T], "(n_metrics, n_plot_axis)"]
SeriesData = Annotated[
    list[
        tuple[
            Annotated[GridsearchResult, "metrics"],
            Annotated[GridsearchResult[np.void], "arg_opts"],
        ]
    ],
    "len=n_grid_axes",
]


class GridsearchResultDetails(TypedDict):
    system: str
    plot_data: list[SavedData]
    series_data: dict[str, SeriesData]
    metrics: list[str]
    grid_params: list[str]
    grid_vals: list[Sequence]
    grid_axes: dict[str, Collection[float]]
    main: float


def diff_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "finitedifference":
        return ps.FiniteDifference
    if normalized_kind == "smoothedfinitedifference":
        return ps.SmoothedFiniteDifference
    elif normalized_kind == "sindy":
        return ps.SINDyDerivative
    else:
        raise ValueError


def feature_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind is None:
        return ps.PolynomialLibrary
    elif normalized_kind == "polynomial":
        return ps.PolynomialLibrary
    elif normalized_kind == "fourier":
        return ps.FourierLibrary
    elif normalized_kind == "weak":
        return ps.WeakPDELibrary
    elif normalized_kind == "pde":
        return ps.PDELibrary
    else:
        raise ValueError


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    elif normalized_kind == "miosr":
        return ps.MIOSR
    elif normalized_kind == "trap":
        return ps.TrappingSR3
    elif normalized_kind == "ensemble":
        return ps.EnsembleOptimizer
    else:
        raise ValueError


def coeff_metrics(coefficients, coeff_true):
    metrics = {}
    metrics["coeff_precision"] = sklearn.metrics.precision_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_recall"] = sklearn.metrics.recall_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_f1"] = sklearn.metrics.f1_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_mse"] = sklearn.metrics.mean_squared_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["coeff_mae"] = sklearn.metrics.mean_absolute_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["main"] = metrics["coeff_f1"]
    return metrics


def integration_metrics(model, x_test, t_train, x_dot_test):
    metrics = {}
    metrics["mse-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_squared_error,
    )
    metrics["mae-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_absolute_error,
    )
    return metrics


def unionize_coeff_matrices(
    model: ps.SINDy, coeff_true: list[dict[str, float]]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Reformat true coefficients and coefficient matrix compatibly

    In order to calculate accuracy metrics between true and estimated
    coefficients, this function compares the names of true coefficients
    and a the fitted model's features in order to create comparable
    (i.e. non-ragged) true and estimated coefficient matrices.  In
    a word, it stacks the correct coefficient matrix and the estimated
    coefficient matrix in a matrix that represents the union of true
    features and modeled features.

    Arguments:
        model: fitted model
        coeff_true: list of dicts of format function_name: coefficient,
            one dict for each modeled coordinate/target

    Returns:
        Tuple of true coefficient matrix, estimated coefficient matrix,
        and combined feature names

    Warning:
        Does not disambiguate between commutatively equivalent function
        names such as 'x z' and 'z x' or 'x^2' and 'x x'
    """
    model_features = model.get_feature_names()
    true_features = [set(coeffs.keys()) for coeffs in coeff_true]
    unmodeled_features = set(chain.from_iterable(true_features)) - set(model_features)
    model_features.extend(list(unmodeled_features))
    est_coeff_mat = model.coefficients()
    new_est_coeff = np.zeros((est_coeff_mat.shape[0], len(model_features)))
    new_est_coeff[:, : est_coeff_mat.shape[1]] = est_coeff_mat
    true_coeff_mat = np.zeros_like(new_est_coeff)
    for row, terms in enumerate(coeff_true):
        for term, coeff in terms.items():
            true_coeff_mat[row, model_features.index(term)] = coeff

    return true_coeff_mat, new_est_coeff, model_features


def make_model(
    input_features: list[str],
    dt: float,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
) -> ps.SINDy:
    """Build a model with object parameters dictionaries

    e.g. {"kind": "finitedifference"} instead of FiniteDifference()
    """

    def finalize_param(lookup_func, pdict, lookup_key):
        try:
            cls_name = pdict.pop(lookup_key)
        except AttributeError:
            cls_name = pdict.vals.pop(lookup_key)
            pdict = pdict.vals

        param_cls = lookup_func(cls_name)
        param_final = param_cls(**pdict)
        pdict[lookup_key] = cls_name
        return param_final

    diff = finalize_param(diff_lookup, diff_params, "diffcls")
    features = finalize_param(feature_lookup, feat_params, "featcls")
    opt = finalize_param(opt_lookup, opt_params, "optcls")
    return ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,  # type: ignore
        feature_library=features,
        feature_names=input_features,
    )


def simulate_test_data(model: ps.SINDy, dt: float, x_test: Float2D) -> SINDyTrialUpdate:
    """Add simulation data to grid_data

    This includes the t_sim and x_sim keys.  Does not mutate argument.
    Returns:
        Complete GridPointData
    """
    t_test = cast(Float1D, np.arange(0, len(x_test) * dt, step=dt))
    t_sim = t_test
    try:
        x_sim = cast(Float2D, model.simulate(x_test[0], t_test))
    except ValueError:
        warn(message="Simulation blew up; returning zeros")
        x_sim = np.zeros_like(x_test)
    # truncate if integration returns wrong number of points
    t_sim = cast(Float1D, t_test[: len(x_sim)])
    return {"t_sim": t_sim, "x_sim": x_sim, "t_test": t_test}


@dataclass
class SeriesDef:
    """The details of constructing the ragged axes of a grid search.

    The concept of a SeriesDef refers to a slice along a single axis of
    a grid search in conjunction with another axis (or axes)
    whose size or meaning differs along different slices.

    Attributes:
        name: The name of the slice, as a label for printing
        static_param: the constant parameter to this slice. Then key is
            the name of the parameter, as understood by the experiment
            Conceptually, the key serves as an index of this slice in
            the gridsearch.
        grid_params: the keys of the parameters in the experiment that
            vary along jagged axis for this slice
        grid_vals: the values of the parameters in the experiment that
            vary along jagged axis for this slice

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )

    """

    name: str
    static_param: dict
    grid_params: Optional[Sequence[str]]
    grid_vals: Optional[list[Sequence]]


@dataclass
class SeriesList:
    """Specify the ragged slices of a grid search.

    As an example, consider a grid search of miles per gallon for
    different vehicles, in different routes, with different tires.
    Since different tires fit on different vehicles, the tire axis would
    be ragged, varying along the vehicle axis.

        Truck = SeriesDef("trucks")

    Attributes:
        param_name: the key of the parameter in the experiment that
            varies along the series axis.
        print_name: the print name of the parameter in the experiment
            that varies along the series axis.
        series_list: Each element of the series axis

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )
        bike_tires = SeriesDef(
            "Bike",
            {"vehicle": "bicycle"},
            ["vehicle.tires"],
            [["gravel_tires", "road_tires"]]
        )
        VehicleOptions = SeriesList(
            "vehicle",
            "Vehicle Types",
            [truck_wheels, bike_tires]
        )

    """

    param_name: Optional[str]
    print_name: Optional[str]
    series_list: list[SeriesDef]


class NestedDict(defaultdict):
    """A dictionary that splits all keys by ".", creating a sub-dict.

    Args: see superclass

    Example:

        >>> foo = NestedDict("a.b"=1)
        >>> foo["a.c"] = 2
        >>> foo["a"]["b"]
        1
    """

    def __missing__(self, key):
        try:
            prefix, subkey = key.split(".", 1)
        except ValueError:
            raise KeyError(key)
        return self[prefix][subkey]

    def __setitem__(self, key, value):
        if "." in key:
            prefix, suffix = key.split(".", 1)
            if self.get(prefix) is None:
                self[prefix] = NestedDict()
            return self[prefix].__setitem__(suffix, value)
        else:
            return super().__setitem__(key, value)

    def update(self, other: dict):  # type: ignore
        try:
            for k, v in other.items():
                self.__setitem__(k, v)
        except:  # noqa: E722
            super().update(other)

    def flatten(self):
        """Flattens a nested dictionary without mutating.  Returns new dict"""

        def _flatten(nested_d: dict) -> dict:
            new = {}
            for key, value in nested_d.items():
                if not isinstance(key, str):
                    raise TypeError("Only string keys allowed in flattening")
                if not isinstance(value, dict):
                    new[key] = value
                    continue
                for sub_key, sub_value in _flatten(value).items():
                    new[key + "." + sub_key] = sub_value
            return new

        return _flatten(self)


def kalman_generalized_cv(
    times: np.ndarray, measurements: np.ndarray, alpha0: float = 1, detail=False
):
    """Find kalman parameter alpha using GCV error

    See Boyd & Barratt, Fitting a Kalman Smoother to Data.  No regularization
    """
    measurements = measurements.reshape((-1, 1))
    dt = times[1] - times[0]
    Ai = np.array([[1, 0], [dt, 1]])
    Qi = kalman.gen_Qi(dt)
    Qi_rt_inv = np.linalg.cholesky(np.linalg.inv(Qi))
    Qi_r_i_vec = np.reshape(Qi_rt_inv, (-1, 1))
    Qi_proj = (
        lambda vec: Qi_r_i_vec
        @ (Qi_r_i_vec.T @ Qi_r_i_vec) ** -1
        @ (Qi_r_i_vec.T)
        @ vec
    )
    Hi = np.array([[0, 1]])
    Ri = np.eye(1)
    Ri_rt_inv = Ri
    params0 = aks.KalmanSmootherParameters(Ai, Qi_rt_inv, Hi, Ri)
    mask = np.ones_like(measurements, dtype=bool)
    mask[::4] = False

    def proj(curr_params, t):
        W_n_s_v = np.reshape(curr_params.W_neg_sqrt, (-1, 1))
        W_n_s_v = np.reshape(Qi_proj(W_n_s_v), (2, 2))
        new_params = aks.KalmanSmootherParameters(Ai, W_n_s_v, Hi, Ri_rt_inv)
        return new_params, t

    params, info = aks.tune(params0, proj, measurements, K=mask, lam=0.1, verbose=False)
    est_Q = np.linalg.inv(params.W_neg_sqrt @ params.W_neg_sqrt.T)
    est_alpha = 1 / (est_Q / Qi).mean()
    return est_alpha


def _amax_to_full_inds(
    amax_inds: Collection[tuple[int | slice, int] | ellipsis] | ellipsis,
    amax_arrays: list[list[GridsearchResult[np.void]]],
) -> set[tuple[int, ...]]:
    """Find full indexers to selected elements of argmax arrays

    Args:
        amax_inds: selection statemtent of which argmaxes to return.
        amax_arrays: arrays of indexes to full gridsearch that are responsible for
            the computed max values.  First level of nesting reflects series(?), second
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


def strict_find_grid_match(
    results: GridsearchResultDetails,
    *,
    params: Optional[dict[str, Any]] = None,
    ind_spec: Optional[tuple[int | slice, int] | ellipsis] = None,
) -> SINDyTrialData:
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


_EqTester = TypeVar("_EqTester")


def _param_normalize(val: _EqTester) -> _EqTester | str:
    if type(val).__eq__ == object.__eq__:
        return repr(val)
    else:
        return val
