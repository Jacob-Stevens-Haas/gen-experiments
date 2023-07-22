
from abc import ABCMeta
from typing import NewType
from typing import Protocol
from typing import Union

import numpy as np

from scipy import sparse

# Attempt 1: Metaclasses
class ArrayLike(metaclass=ABCMeta): ...


ArrayLike.register(sparse.spmatrix)
ArrayLike.register(np.ndarray)

CovarianceMatrix = NewType("CovarianceMatrix", ArrayLike)
std_cov = CovarianceMatrix(np.eye(2))
nearly_singular_cov = CovarianceMatrix(np.array([[1,1],[1, 1.1]]))


def mahalanobis_distance(x: ArrayLike, mu: ArrayLike, cov: CovarianceMatrix) -> float:
    """Calculate standardized distance of a random vector from its mean"""
    centered = x-mu
    return centered.T @ cov @ centered


mahalanobis_distance(np.zeros(2), np.zeros(2), nearly_singular_cov)

# Attempt 2: Protocol
class ArrayProtocol(Protocol):
    def __matmul__(self, other): ...


InequalityConstraint = NewType("InequalityConstraint", ArrayProtocol)
threshold = InequalityConstraint(np.arange(4))


def cone_indicator(
    point: ArrayProtocol,
    cone_origin: InequalityConstraint
) -> float:
    """Evaluate indicator function for the positive orthogonal cone"""
    return 0 if (point - cone_origin > 0).all() else float("inf")


cone_indicator(np.zeros(4), threshold)

# Attempt 3: Union
AnyArray = Union[sparse.spmatrix, np.ndarray]
ToeplitzMatrix = NewType("ToeplitzMatrix", AnyArray)
boring_array = ToeplitzMatrix(np.eye(3))


def list_diagonals(arr: ToeplitzMatrix):
    return np.concatenate((arr[:, 0].ravel(), arr[0,1:].ravel()))


list_diagonals(boring_array)