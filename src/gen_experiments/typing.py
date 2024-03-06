from typing import TypeVar

import numpy as np
from numpy.typing import NBitBase

NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Shape = TypeVar("Shape", bound=tuple[int, ...])
FloatND = np.ndarray[Shape, np.dtype[np.floating[NBitBase]]]
