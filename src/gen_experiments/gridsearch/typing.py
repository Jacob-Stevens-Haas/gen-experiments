from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import EllipsisType as ellipsis
from typing import Annotated, Any, Collection, Optional, Sequence, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GridLocator:
    """A specification of which points in a gridsearch to match.

    Rather than specifying the exact point in the mega-grid of every
    varied axis, specify by result, e.g "all of the points from the
    Kalman series that had the best mean squared error as noise was
    varied.

    Args:
        metric: The metric in which to find results.  An ellipsis means "any metrics"
        keep_axis: The grid-varied parameter in which to find results, or a tuple of
            that axis and position along that axis.  To search a particular value of
            that parameter, use the param_match kwarg.  An ellipsis means "any axis"
        param_match: A collection of dictionaries to match parameter values represented
            by points in the gridsearch.  Dictionary equality is checked for every
            non-callable value; for callable values, it is applied to the grid
            parameters and must return a boolean.  Logical OR is applied across the
            collection
    """

    metric: str | ellipsis = field(default=...)
    keep_axis: str | tuple[str, int] | ellipsis = field(default=...)
    param_match: Collection[dict[str, Any]] = field(default=())


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

ExpResult = dict[str, Any]


class SavedGridPoint(TypedDict):
    params: dict
    pind: tuple[int]
    data: ExpResult


class GridsearchResultDetails(TypedDict):
    system: str
    plot_data: list[SavedGridPoint]
    series_data: dict[str, SeriesData]
    metrics: list[str]
    grid_params: list[str]
    grid_vals: list[Sequence]
    grid_axes: dict[str, Collection[float]]
    main: float


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
    grid_params: list[str]
    grid_vals: list[Iterable]


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
