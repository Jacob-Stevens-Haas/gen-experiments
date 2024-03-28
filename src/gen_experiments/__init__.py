import importlib
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pysindy import BaseDifferentiation, FiniteDifference, SINDy

from . import gridsearch, odes, pdes
from .utils import SINDyTrialData, make_model  # noqa: F401

this_module = importlib.import_module(__name__)
BORING_ARRAY = np.ones((2, 2), dtype=float)

Scores = Mapping[str, float]


class FakeModel(SINDy):
    differentiation_method: BaseDifferentiation

    def __init__(self) -> None:
        self.differentiation_method = FiniteDifference()
        self.differentiation_method.smoothed_x_ = np.ones((1, 2))

    def print(self, *args: Any, **kwargs: Any) -> None:
        print("fake model")

    def simulate(self, *args: Any, **kwargs: Any) -> NDArray[np.float64]:
        return BORING_ARRAY


class NoExperiment:
    metric_ordering: dict[str, str] = defaultdict(lambda: "max")
    name = "No Experiment"
    lookup_dict = {"arg": {"foo": 1}}

    @staticmethod
    def run(
        *args: Any, return_all: bool = True, **kwargs: Any
    ) -> Scores | tuple[Scores, SINDyTrialData]:
        metrics = defaultdict(
            lambda: 1,
            main=1,
        )
        if return_all:
            trial_data: SINDyTrialData = {
                "dt": 1,
                "coeff_true": BORING_ARRAY[:1],
                "coeff_fit": BORING_ARRAY[:1],
                # "coefficients": boring_array,
                "feature_names": ["1"],
                "input_features": ["x", "y"],
                "t_train": np.arange(0, 1, 1, dtype=float),
                "x_train": BORING_ARRAY,
                "x_true": BORING_ARRAY,
                "smooth_train": BORING_ARRAY,
                "x_test": BORING_ARRAY,
                "x_dot_test": BORING_ARRAY,
                # "x_train_true": [boring_array],
                "model": FakeModel(),
            }
            return (
                metrics,
                trial_data,
            )
        return metrics


experiments: dict[str, tuple[Any, str | None]] = {
    "sho": (odes, "sho"),
    "lorenz": (odes, "lorenz"),
    "lorenz_2d": (odes, "lorenz_2d"),
    "pendulum": (odes, "pendulum"),
    "cubic_ho": (odes, "cubic_ho"),
    "vdp": (odes, "vdp"),
    "hopf": (odes, "hopf"),
    "duff": (odes, "duff"),
    "lv": (odes, "lv"),
    "ross": (odes, "ross"),
    "gridsearch": (gridsearch, None),
    "diffuse1D_dirichlet": (pdes, "diffuse1D_dirichlet"),
    "diffuse1D_periodic": (pdes, "diffuse1D_periodic"),
    "burgers1D_dirichlet": (pdes, "burgers1D_dirichlet"),
    "burgers1D_periodic": (pdes, "burgers1D_periodic"),
    "ks_dirichlet": (pdes, "ks_dirichlet"),
    "ks_periodic": (pdes, "ks_periodic"),
    "none": (NoExperiment, None),
}
