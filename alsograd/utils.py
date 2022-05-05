from typing import Tuple, Union, Iterable, Any, List, Sequence
import numpy as np


Order = Union[None, Sequence[int]]
Axis = Union[int, Order]


# Backwards pass through np broadcasting
def rev_sum(x: np.ndarray, shape: Tuple[int]):
    axis = [0] if shape == (1, ) else [i for i, s in enumerate(shape) if s == 1 and x.shape[i] > 1]
    return x.sum(axis=tuple(axis)).reshape(shape) if len(axis) > 0 else x


# Maintain shape for backwards pass through reduce operations
def shape_for_keepdims(in_shape: Sequence[int], axis: Axis) -> List[int]:
    ndim = len(in_shape)
    if axis is None:
        axis = range(ndim)

    if not isinstance(axis, Iterable):
        axis = [axis]

    axis = tuple([(ax + ndim) % ndim for ax in axis])
    out_shape = [s for i, s in enumerate(in_shape) if i not in axis]
    return out_shape if len(out_shape) else [1]


def plural(x: Union[Sequence[Any], Any]) -> Tuple[Any, ...]:
    if isinstance(x, Sequence):
        return tuple(x)

    return (x, )
