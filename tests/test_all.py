import numpy as np
import pysindy as ps
import pytest

from gen_experiments.typing import NestedDict
from gen_experiments.utils import unionize_coeff_matrices


def test_flatten_nested_dict():
    deep = NestedDict(a=NestedDict(b=1))
    result = deep.flatten()
    assert deep != result
    expected = {"a.b": 1}
    assert result == expected


def test_flatten_nested_bad_dict():
    nested = {1: NestedDict(b=1)}
    # Testing the very thing that causes a typing error, thus ignoring
    with pytest.raises(TypeError, match="keywords must be strings"):
        NestedDict(**nested)  # type: ignore
    with pytest.raises(TypeError, match="Only string keys allowed"):
        deep = NestedDict(a={1: 1})
        deep.flatten()


def test_unionize_coeff_matrices():
    model = ps.SINDy(feature_names=["x", "y"])
    data = np.arange(10)
    data = np.vstack((data, data)).T
    model.fit(data, 0.1)
    coeff_true = [{"y": -1, "zorp_x": 0.1}, {"x": 1, "zorp_y": 0.1}]
    true, est, feats = unionize_coeff_matrices(model, coeff_true)
    assert len(feats) == true.shape[1]
    assert len(feats) == est.shape[1]
    assert est.shape == true.shape
