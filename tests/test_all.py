import numpy as np
import pytest

import gen_experiments
from gen_experiments import gridsearch
from gen_experiments import utils


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
    result = set(gridsearch._ndindex_skinny((2, 2, 2), (0, 2), ((0,), (-1,))))
    expected = {
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
    }
    assert result == expected


def test_thin_indexing_default():
    result = set(gridsearch._ndindex_skinny((2, 2, 2), (0, 2), None))
    expected = {
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
    }
    assert result == expected


def test_thin_indexing_callable():
    result = set(gridsearch._ndindex_skinny((2, 2, 2), (0, 2), ((0,), (lambda x: x,))))
    expected = {
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
    }
    assert result == expected


def test_curr_skinny_specs():
    grid_params = ["a", "c", "e", "f"]
    skinny_specs = (
        ("a", "b", "c", "d", "e"),
        ((1, 2, 3, 4), (0, 2, 3, 4), (0, 1, 3, 4), (0, 1, 2, 4), (0, 1, 2, 3)),
    )
    ind_skinny, where_others = gridsearch._curr_skinny_specs(skinny_specs, grid_params)
    assert ind_skinny == [0, 1, 2]
    assert where_others == ((2, 4), (0, 4), (0, 2))


def test_marginalize_grid_views():
    results = np.arange(120).reshape(2, 3, 4, 5) # (metrics, param1, param2, param3)
    results[0,0,0,0] = 1000
    grid_decisions = ["plot", "max", "plot"]
    result = gridsearch._marginalize_grid_views(grid_decisions, results)
    assert len(result) == len([dec for dec in grid_decisions if dec =="plot"])
    expected = [
        np.array([[1000, 39, 59], [79, 99, 119]]),
        np.array([[1000, 56, 57, 58, 59], [115, 116, 117, 118, 119]]),
    ]
    assert all((res == ex).all() for res, ex in zip(result, expected))


def test_tuple_argmax():
    arr = np.arange(120).reshape(2, 3, 4, 5)
    arr[0,0,0,0] = 1000
    result = utils._argmax(arr, axis=(1,2))
    expected = np.empty((2, 5))
    expected[:] = 11
    expected[0, 0] = 0
    assert np.array_equal(result, expected)