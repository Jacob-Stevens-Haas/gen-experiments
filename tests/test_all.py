import numpy as np
import pytest

import gen_experiments.gridsearch.typing
from gen_experiments import gridsearch


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
    arr = np.arange(16, dtype=np.float_).reshape(
        2, 2, 2, 2
    )  # (metrics, param1, param2, param3)
    arr[0, 0, 0, 0] = 1000
    arr[-1, -1, -1, 0] = -1000
    arr[0, 0, 0, 1] = np.nan
    grid_decisions = ["plot", "max", "plot"]
    opts = ["max", "min"]
    res_val, res_ind = gridsearch._marginalize_grid_views(grid_decisions, arr, opts)
    assert len(res_val) == len([dec for dec in grid_decisions if dec == "plot"])
    expected_val = [
        np.array([[1000, 7], [8, -1000]]),
        np.array([[1000, 7], [-1000, 9]]),
    ]
    for result, expected in zip(res_val, expected_val):
        np.testing.assert_array_equal(result, expected)

    ts = "i,i,i,i"
    expected_ind = [
        np.array([[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 0, 0, 0), (1, 1, 1, 0)]], ts),
        np.array([[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 1, 1, 0), (1, 0, 0, 1)]], ts),
    ]
    for result, expected in zip(res_ind, expected_ind):
        np.testing.assert_array_equal(result, expected)


def test_argopt_tuple_axis():
    arr = np.arange(16, dtype=np.float_).reshape(2, 2, 2, 2)
    arr[0, 0, 0, 0] = 1000
    result = gridsearch._argopt(arr, (1, 3))
    expected = np.array(
        [[(0, 0, 0, 0), (0, 1, 1, 1)], [(1, 1, 0, 1), (1, 1, 1, 1)]], dtype="i,i,i,i"
    )
    np.testing.assert_array_equal(result, expected)


def test_argopt_empty_tuple_axis():
    arr = np.arange(4, dtype=np.float_).reshape(4)
    result = gridsearch._argopt(arr, ())
    expected = np.array([(0,), (1,), (2,), (3,)], dtype=[("f0", "i")])
    np.testing.assert_array_equal(result, expected)
    result = gridsearch._argopt(arr, None)
    pass


def test_argopt_int_axis():
    arr = np.arange(8, dtype=np.float_).reshape(2, 2, 2)
    arr[0, 0, 0] = 1000
    result = gridsearch._argopt(arr, 1)
    expected = np.array([[(0, 0, 0), (0, 1, 1)], [(1, 1, 0), (1, 1, 1)]], dtype="i,i,i")
    np.testing.assert_array_equal(result, expected)


def test_index_in():
    match_me = (1, ..., slice(None), 3)
    good = [(1, 2, 1, 3), (1, 1, 3)]
    for g in good:
        assert gridsearch._index_in(g, match_me)
    bad = [(1, 3), (1, 1, 2), (1, 1, 1, 2)]
    for b in bad:
        assert not gridsearch._index_in(b, match_me)


def test_index_in_errors():
    with pytest.raises(ValueError):
        gridsearch._index_in((1,), (slice(-1),))


def test_flatten_nested_dict():
    deep = gen_experiments.gridsearch.typing.NestedDict(
        a=gen_experiments.gridsearch.typing.NestedDict(b=1)
    )
    result = deep.flatten()
    assert deep != result
    expected = {"a.b": 1}
    assert result == expected


def test_grid_locator_match():
    m_params = {"sim_params.t_end": 10, "foo": 1}
    m_ind = (0, 1)
    # Effectively testing the clause: (x OR y OR ...) AND (a OR b OR ...)
    # Note: OR() with no args is falsy
    # also note first index is stripped ind_spec
    good_specs = [
        (({"sim_params.t_end": 10},), ((1, 0, 1),)),
        (({"sim_params.t_end": 10},), ((1, 0, 1), (1, 0, ...))),
        (({"sim_params.t_end": 10}, {"foo": 1}), ((1, 0, 1),)),
        (({"sim_params.t_end": 10}, {"bar: 1"}), ((1, 0, 1),)),
        (
            ({"sim_params.t_end": 10},),
            (
                (1, 0, 1),
                (
                    1,
                    1,
                ),
            ),
        ),
    ]
    for param_spec, ind_spec in good_specs:
        assert gridsearch._grid_locator_match(m_params, m_ind, param_spec, ind_spec)

    bad_specs = [
        ((), ((0, 1),)),
        (({"sim_params.t_end": 10},), ()),
        (({"sim_params.t_end": 9},), ((1, 0, 1),)),
        (({"sim_params.t_end": 10},), ((1, 0, 0),)),
        ((), ()),
    ]
    for param_spec, ind_spec in bad_specs:
        assert not gridsearch._grid_locator_match(m_params, m_ind, param_spec, ind_spec)


def test_amax_to_full_inds():
    amax_inds = ((1, 1), (slice(None), 0))
    arr = np.array([[(0, 0), (0, 1)], [(1, 0), (1, 1)]], dtype="i,i")
    amax_arrays = [[arr, arr], [arr]]
    result = gridsearch._amax_to_full_inds(amax_inds, amax_arrays)
    expected = {(0, 0), (1, 1), (1, 0)}
    assert result == expected
    result = gridsearch._amax_to_full_inds(..., amax_arrays)
    expected |= {(0, 1)}
    return result == expected


def test_flatten_nested_bad_dict():
    with pytest.raises(TypeError, match="keywords must be strings"):
        gen_experiments.gridsearch.typing.NestedDict(
            **{1: gen_experiments.gridsearch.typing.NestedDict(b=1)}
        )
    with pytest.raises(TypeError, match="Only string keys allowed"):
        deep = gen_experiments.gridsearch.typing.NestedDict(a={1: 1})
        deep.flatten()
