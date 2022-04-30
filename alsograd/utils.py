from typing import Tuple, Union, Iterable, Any
import numpy as np


# Backwards pass through sum operation
def rev_sum(x: np.ndarray, shape: Tuple[int]):
    axis = [0] if shape == (1, ) else [i for i, s in enumerate(shape) if s == 1 and x.shape[i] > 1]
    return x.sum(axis=axis).reshape(shape) if len(axis) > 0 else x


def plural(x: Union[Iterable[Any], Any]) -> Tuple[Any, ...]:
    return x if isinstance(x, tuple) else (x, )
