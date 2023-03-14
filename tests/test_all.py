import numpy as np

import gen_experiments


def test_enumerate_grid():
    arr = np.arange(120).reshape(1, 2, 3, 4, 5)
    grid_results = gen_experiments.gridsearch._marginalize_grid_views(
        ["plot", "plot", "max", "max"], arr
    )
    grid_expected = [np.array([[59, 119]]), np.array([[79, 99, 119]])]
    assert len(grid_results) == len(grid_expected)
    for result, expected in zip(grid_results, grid_expected):
        np.testing.assert_array_equal(result, expected)
