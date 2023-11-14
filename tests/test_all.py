import numpy as np
import pytest

import gen_experiments
from gen_experiments import gridsearch
from gen_experiments import utils


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
    arr = np.arange(120).reshape(2, 3, 4, 5) # (metrics, param1, param2, param3)
    arr[0,0,0,0] = 1000
    grid_decisions = ["plot", "max", "plot"]
    result_val, result_ind = gridsearch._marginalize_grid_views(grid_decisions, arr)
    assert len(result_val) == len([dec for dec in grid_decisions if dec =="plot"])
    expected_val = [
        np.array([[1000, 39, 59], [79, 99, 119]]),
        np.array([[1000, 56, 57, 58, 59], [115, 116, 117, 118, 119]]),
    ]
    for result, expected in zip(result_val, expected_val):
        np.testing.assert_array_equal(result, expected)

    expected_ind = [
        np.array([[0, 19, 19], [19, 19, 19]]),
        np.array([[0, 11, 11, 11, 11], [11, 11, 11, 11, 11]])
    ]
    for result, expected in zip(result_ind, expected_ind):
        np.testing.assert_array_equal(result, expected)


def test_tuple_argmax():
    arr = np.arange(120).reshape(2, 3, 4, 5)
    arr[0,0,0,0] = 1000
    result = utils._argmax(arr, axis=(1,2))
    expected = np.empty((2, 5))
    expected[:] = 11
    expected[0, 0] = 0
    assert np.array_equal(result, expected)

def test_flatten_nested_dict():
    deep = utils.NestedDict(a=utils.NestedDict(b=1))
    result = deep.flatten()
    assert deep != result
    expected = {"a.b": 1}
    assert result == expected

def test_flatten_nested_bad_dict():
    with pytest.raises(TypeError, match="keywords must be strings"):
        utils.NestedDict(**{1: utils.NestedDict(b=1)})
    with pytest.raises(TypeError, match="Only string keys allowed"):
        deep = utils.NestedDict(a={1: 1})
        deep.flatten()
