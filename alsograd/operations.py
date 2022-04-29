from __future__ import annotations

from typing import Optional, Union, Tuple, TYPE_CHECKING
import numpy as np

from alsograd.utils import rev_sum
from alsograd.core import Operation


class Add(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape, b.shape)
        return a + b

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_shape, b_shape = self.cache
        return rev_sum(g, a_shape), rev_sum(g, b_shape)


class Sub(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape, b.shape)
        return a - b

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_shape, b_shape = self.cache
        return rev_sum(g, a_shape), rev_sum(-g, b_shape)


class Mul(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a, b)
        return a*b

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.cache
        return rev_sum(g*a, a.shape), rev_sum(g*b, b.shape)


# Reduce operations
class Sum(Operation):
    def __init__(self, axis: Optional[int]=None):
        self.axis = axis

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape)
        return a.sum(axis=self.axis, keepdims=True)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a_shape, = self.cache
        return np.broadcast_to(g, a_shape)

