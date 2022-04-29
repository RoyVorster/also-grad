from .core import *
from .utils import *

import numpy as np


class Add(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape, b.shape)
        return a + b

    def backward(self, g: np.ndarray) -> [Parameter, Parameter]:
        a_shape, b_shape = self.cache
        return rev_sum(g, a_shape), rev_sum(g, b_shape)


class Mul(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a, b)
        return a*b

    def backward(self, g: np.ndarray) -> [Parameter, Parameter]:
        a, b = self.cache
        return rev_sum(g*a, a.shape), rev_sum(g*b, b.shape)

class Sum(Operation):
    def forward(self, a: np.ndarray, axis=None) -> np.ndarray:
        self.add_to_cache(a.shape)
        return a.sum(axis=axis, keepdims=True)

    def backward(self, g: np.ndarray) -> Parameter:
        a_shape, = self.cache
        return np.broadcast_to(g, a_shape)

