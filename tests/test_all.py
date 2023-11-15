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
    arr = np.arange(16).reshape(2, 2, 2, 2) # (metrics, param1, param2, param3)
    arr[0,0,0,0] = 1000
    grid_decisions = ["plot", "max", "plot"]
    result_val, result_ind = gridsearch._marginalize_grid_views(grid_decisions, arr)
    assert len(result_val) == len([dec for dec in grid_decisions if dec =="plot"])
    expected_val = [
        np.array([[1000, 7], [11, 15]]),
        np.array([[1000, 7], [14, 15]]),
    ]
    for result, expected in zip(result_val, expected_val):
        np.testing.assert_array_equal(result, expected)

    ts = "i,i,i,i"
    expected_ind = [
        np.array([[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 0, 1, 1), (1, 1, 1, 1)]], ts),
        np.array([[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 1, 1, 0), (1, 1, 1, 1)]], ts)
    ]
    for result, expected in zip(result_ind, expected_ind):
        np.testing.assert_array_equal(result, expected)

def test_argmax_tuple_axis():
    arr = np.arange(16).reshape(2, 2, 2, 2)
    arr[0,0,0,0] = 1000
    result = utils._argmax(arr, (1, 3))
    expected = np.array(
        [[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 1, 0, 1), (1, 1, 1, 1)]],
        dtype="i,i,i,i"
    )
    np.testing.assert_array_equal(result, expected)


def test_argmax_int_axis():
    arr = np.arange(8).reshape(2, 2, 2)
    arr[0,0,0] = 1000
    result = utils._argmax(arr, 1)
    expected = np.array(
        [[(0, 0, 0), (0, 1, 1)], [(1, 1, 0), (1, 1, 1)]], dtype="i,i,i"
    )
    np.testing.assert_array_equal(result, expected)


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
