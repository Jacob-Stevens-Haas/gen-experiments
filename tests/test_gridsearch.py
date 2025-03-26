import numpy as np
import pysindy as ps
import pytest

from gen_experiments import gridsearch
from gen_experiments.gridsearch.typing import (
    GridsearchResultDetails,
    SavedGridPoint,
    SeriesData,
)


def test_thick_indexing():
    result = set(gridsearch._ndindex_skinny((2,), None, None))
    expected = {(0,), (1,)}
    assert result == expected


def test_thick_indexing_alt():
    result = set(gridsearch._ndindex_skinny((2,), [], None))
    expected = {(0,), (1,)}
    assert result == expected


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

    ts = "i,i,i"
    expected_ind = [
        np.array([[(0, 0, 0), (1, 1, 1)], [(0, 0, 0), (1, 1, 0)]], ts),
        np.array([[(0, 0, 0), (1, 1, 1)], [(1, 1, 0), (0, 0, 1)]], ts),
    ]
    for result, expected in zip(res_ind, expected_ind):
        np.testing.assert_array_equal(result, expected)


def test_marginalize_grid_views_noplot():
    arr = np.array(([[0.0, 1.0], [2.0, 3.0]]))  # (metrics, param1)
    grid_decisions = ["max"]
    opts = ["max", "min"]
    res_val, res_ind = gridsearch._marginalize_grid_views(grid_decisions, arr, opts)
    assert len(res_val) == 1
    expected_val = [
        np.array([[1.0], [2.0]]),
    ]
    for result, expected in zip(res_val, expected_val):
        np.testing.assert_array_equal(result, expected)

    ts = np.dtype([("f0", "<i4")])
    expected_ind = [
        np.array([[(1,)], [(0,)]], ts),
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


@pytest.fixture
def gridsearch_results():
    want: SavedGridPoint = {
        "params": {"diff_params.alpha": 0.1, "opt_params": ps.STLSQ()},
        "pind": (1,),
        "data": {},
    }
    dont_want: SavedGridPoint = {
        "params": {"diff_params.alpha": 0.2, "opt_params": ps.SSR()},
        "pind": (0,),
        "data": {},
    }
    tup_dtype = np.dtype([("f0", "i")])
    max_amax: SeriesData = [
        (np.ones((2, 2)), np.array([[(1,), (0)], [(0,), (0,)]], dtype=tup_dtype)),
        (np.ones((2, 2)), np.array([[(0,), (0,)], [(0,), (0,)]], dtype=tup_dtype)),
    ]
    full_details: GridsearchResultDetails = {
        "system": "sho",
        "plot_data": [want, dont_want],
        "series_data": {"foo": max_amax},
        "metrics": ("mse", "mae"),
        "scan_grid": {"sim_params.t_end": [1, 2], "sim_params.noise": [5, 6]},
        "plot_grid": {},
        "grid_params": ["sim_params.t_end", "bar", "sim_params.noise"],
        "grid_vals": [[1, 2], [7, 8], [5, 6]],
        "main": 1,
    }
    return want, full_details


@pytest.mark.parametrize(
    "locator",
    (
        gridsearch.GridLocator(
            ("mse",), (("sim_params.t_end",), ...), [{"diff_params.alpha": 0.1}]
        ),
        gridsearch.GridLocator(
            ("mse",), (("sim_params.t_end",), ...), [{"opt_params": ps.STLSQ()}]
        ),
        gridsearch.GridLocator(
            ..., (..., ...), [{"diff_params.alpha": lambda x: x < 0.2}]
        ),
        gridsearch.GridLocator(("mse",), {("sim_params.t_end", (0,))}, [{}]),
        gridsearch.GridLocator(
            ..., (..., ...), [{"diff_params.alpha": 0.1}, {"diff_params.alpha": 0.3}]
        ),
        gridsearch.GridLocator(params_or=[{"diff_params.alpha": 0.1}, {"foo": 0}]),
    ),
    ids=("exact", "object", "callable", "by_axis", "or", "missingkey"),
)
def test_find_gridpoints(gridsearch_results, locator):
    want, full_details = gridsearch_results
    results = gridsearch.find_gridpoints(
        locator,
        full_details["plot_data"],
        full_details["series_data"].values(),
        full_details["metrics"],
        full_details["scan_grid"],
    )
    assert [want] == results


def test_grid_locator_match():
    m_params = {"sim_params.t_end": 10, "foo": 1}
    m_ind = (0, 1)
    # Effectively testing the clause: (x OR y OR ...) AND (a OR b OR ...)
    # Note: OR() with no args is falsy, AND() with no args is thruthy
    # also note first index is stripped ind_spec
    good_specs = [
        (({"sim_params.t_end": 10},), ((1, 0, 1),)),
        (({"sim_params.t_end": 10},), ((1, 0, 1), (1, 0, ...))),
        (({"sim_params.t_end": 10}, {"foo": 1}), ((1, 0, 1),)),
        (({"sim_params.t_end": 10}, {"bar: 1"}), ((1, 0, 1),)),
        (({"sim_params.t_end": 10},), ((1, 0, 1), (1, 1))),
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


def test_gridsearch_mock():
    results = gridsearch.run(
        1,
        "none",
        grid_params=["foo"],
        grid_vals=[[0, 1]],
        grid_decisions=["plot"],
        other_params={"bar": False, "sim_params": {}},
        metrics=("mse", "mae"),
    )
    assert len(results["plot_data"]) == 0


def test_gridsearch_mock_noplot():
    results = gridsearch.run(
        1,
        "none",
        grid_params=["foo"],
        grid_vals=[[0, 1]],
        grid_decisions=["max"],
        other_params={"bar": False, "sim_params": {}},
        metrics=("mse", "mae"),
    )
    assert len(results["plot_data"]) == 0
