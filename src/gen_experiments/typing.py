from collections import defaultdict
from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import NBitBase

NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Shape = TypeVar("Shape", bound=tuple[int, ...])
FloatND = np.ndarray[Shape, np.dtype[np.floating[NBitBase]]]


class ProbData(NamedTuple):
    dt: float
    t_train: Float1D
    x_train: list[FloatND]
    x_test: list[FloatND]
    x_dot_test: list[FloatND]
    x_train_true: list[FloatND]
    x_train_true_dot: list[FloatND]
    input_features: list[str]
    coeff_true: list[dict[str, float]]


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
