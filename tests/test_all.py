import pytest

from gen_experiments.typing import NestedDict


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
