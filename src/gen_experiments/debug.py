from typing import Annotated, Generic, TypedDict, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NBitBase, NDArray

# T = TypeVar("T")

# class Foo[T]:
#     items: list[T]

#     def __init__(self, thing: T):
#         self.items = [thing, thing]

# Bar =


T = TypeVar("T", bound=np.generic)
Foo = NDArray[T]
Bar = Annotated[NDArray, "foobar"]

lil_foo = NDArray[np.void]


def baz(qux: Foo[np.void]):
    pass
