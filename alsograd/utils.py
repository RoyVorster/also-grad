from typing import Tuple, Union, Iterable, Any, List, Sequence
import numpy as np


Axis = Union[None, Sequence[int]]


# Backwards pass through np broadcasting
def rev_sum(x: np.ndarray, shape: Tuple[int]):
    axis = [0] if shape == (1, ) else [i for i, s in enumerate(shape) if s == 1 and x.shape[i] > 1]
    return x.sum(axis=tuple(axis)).reshape(shape) if len(axis) > 0 else x


def axis_for_keepdims(in_shape: Sequence[int], axis: Axis) -> Tuple[Tuple[int, ...], List[int]]:
    ndim = len(in_shape)
    if axis is None:
        axis = range(ndim)

    if not isinstance(axis, Iterable):
        axis = [axis]

    axis = tuple([(ax + ndim) % ndim for ax in axis])
    out_shape = [s for i, s in enumerate(in_shape) if i not in axis]
    return axis, out_shape if len(out_shape) else [1]


def plural(x: Union[Iterable[Any], Any]) -> Tuple[Any, ...]:
    return x if isinstance(x, tuple) else (x, )
