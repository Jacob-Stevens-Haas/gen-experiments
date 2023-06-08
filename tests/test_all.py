import numpy as np
import pytest

import gen_experiments
from gen_experiments import gridsearch

def test_enumerate_grid():
    arr = np.arange(120).reshape(1, 2, 3, 4, 5)
    grid_results = gen_experiments.gridsearch._marginalize_grid_views(
        ["plot", "plot", "max", "max"], arr
    )
    grid_expected = [np.array([[59, 119]]), np.array([[79, 99, 119]])]
    assert len(grid_results) == len(grid_expected)
    for result, expected in zip(grid_results, grid_expected):
        np.testing.assert_array_equal(result, expected)


def test_thin_indexing():
    result = set(gridsearch._ndindex_skinny((2,2,2), (0,2), ((0,),(-1,))))
    expected = {
        (0,0,0),
        (0,1,0),
        (1,0,0),
        (1,1,0),
        (1,0,1),
        (1,1,1),
    }
    assert result == expected

def test_thin_indexing_default():
    result = set(gridsearch._ndindex_skinny((2,2,2), (0,2), None))
    expected = {
        (0,0,0),
        (0,1,0),
        (1,0,0),
        (1,1,0),
        (0,0,1),
        (0,1,1),
    }
    assert result == expected

def test_thin_indexing_callable():
    result = set(gridsearch._ndindex_skinny((2,2,2), (0,2), ((0,), (lambda x: x,))))
    expected = {
        (0,0,0),
        (0,1,0),
        (1,0,0),
        (1,1,0),
        (1,0,1),
        (1,1,1),
    }
    assert result == expected


def test_plot_param_match():
    plot_prefs = {"sim_params.t_end": 20}
    exp_params = {"sim_params"}
    _params_match(exp_params, plot_prefs)