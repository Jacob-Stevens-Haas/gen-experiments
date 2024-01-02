import importlib
from collections import defaultdict
from warnings import warn

import numpy as np

from mitosis import Parameter

from . import odes
from . import pdes
from . import gridsearch
from . import config

this_module = importlib.import_module(__name__)


class NoExperiment:
    metric_ordering = defaultdict(lambda: "max")
    name = "No Experiment"
    lookup_dict = {"arg": {"foo": 1}}
    @staticmethod
    def run(*args, return_all=True, **kwargs):
        boring_array = np.ones((2, 2))
        metrics = {"main": 1, **defaultdict(lambda: 1)}
        if return_all:
            return (
                metrics,
                {
                    "dt": 1,
                    "coeff_true": boring_array,
                    "coefficients": boring_array,
                    "feature_names": ["1"],
                    "input_features": ["x", "y"],
                    "t_train": np.arange(0, 1, 1),
                    "x_train": [boring_array],
                    "x_test": [boring_array],
                    "x_dot_test": [boring_array],
                    "x_train_true": [boring_array],
                    "model": type(
                        "FakeModel",
                        (),
                        {
                            "print": lambda self: print("fake model"),
                            "simulate": lambda self, x0, ts: boring_array,
                            "differentiation_method": type(
                                "FakeDiff", (), {"smoothed_x_": np.ones((1, 2))}
                            )(),
                        },
                    )(),
                },
            )
        return metrics


experiments = {
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
    "diffuse1D": (pdes, "diffuse1D"),
    "burgers1D": (pdes, "burgers1D"),
    "ks": (pdes, "ks"),
    "none": (NoExperiment, None),
}
ex_name = type("identidict", (), {"__getitem__": lambda self, key: key})()


def lookup_params(params: list[str], config_dict: dict=None) -> list[Parameter]:
    def create_parameter(choice):
        try:
            vals = choice.vals
            modules = choice.modules
        except AttributeError:
            vals = choice
            modules = []
        return vals, modules
    def lookup(arg_name, variant_name):
        if config_dict:
            try:
                return config_dict[arg_name][variant_name]
            except KeyError:
                pass
                warn(
                    f"Could not locate [{arg_name}][{variant_name}] in config_dict",
                    stacklevel=2
                )
        else:
            warn("No configuration dictionary passed.", DeprecationWarning)
        return vars(config)[arg_name][variant_name]

    resolved_params = []
    for param in params:
        arg_name, variant_name = param.split("=")
        vals, modules = create_parameter(lookup(arg_name, variant_name))
        resolved_params.append(Parameter(variant_name, arg_name, vals, modules))

    return resolved_params
