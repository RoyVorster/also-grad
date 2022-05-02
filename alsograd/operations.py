from typing import Optional, Tuple, Union, Sequence, Any
import numpy as np

from alsograd.utils import rev_sum
from alsograd.core import Operation, OperationSimple


Axis = Union[None, Sequence[int]]


class Pow(Operation):
    def __init__(self, exp: float) -> None:
        self.exp = exp

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a)
        return a**self.exp

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, = self.cache
        return g*a*self.exp**(self.exp - 1)


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


class Div(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.add_to_cache(a, b)
        return a/b

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.cache
        return rev_sum(g/b, a.shape), rev_sum(-g*a/(b**2), b.shape)


# Reduce operations
class Sum(Operation):
    def __init__(self, axis: Axis = None):
        self.axis = axis

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape)
        return a.sum(axis=self.axis, keepdims=True)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a_shape, = self.cache
        return np.broadcast_to(g, a_shape)


class Max(Operation):
    def __init__(self, axis: Optional[int] = None):
        self.axis = axis

    def forward(self, a: np.ndarray) -> np.ndarray:
        out = np.amax(a, axis=self.axis)
        self.add_to_cache(a, out)
        return out

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, out = self.cache
        flow = (a == out)
        return flow*g/flow.sum()


# LA operations
class Dot(Operation):
    def forward(self, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.add_to_cache(a, w)
        return a@w

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, w = self.cache
        return g@w.T, a.T@g


class Reshape(Operation):
    def __init__(self, *shape: int) -> None:
        self.shape = shape

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a.shape)
        return a.reshape(self.shape)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a_shape, = self.cache
        return g.reshape(a_shape)


class Transpose(Operation):
    def __init__(self, axis: Axis) -> None:
        self.axis = axis

    def forward(self, a: np.ndarray) -> np.ndarray:
        return np.transpose(a, self.axis)

    def backward(self, g: np.ndarray) -> np.ndarray:
        axis = np.argsort(self.axis) if self.axis is not None else None
        return np.transpose(g, axis)


class Slice(Operation):
    def __init__(self, key: Any) -> None:
        self.key = key

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a)
        return a[self.key]

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, = self.cache
        g_out = np.zeros_like(a)
        g_out[self.key] = g

        return g_out


# Simple operations
def Neg():
    return OperationSimple(lambda x: -1*x, lambda x: -1*x)


def Log():
    return OperationSimple(np.log, lambda x: 1/x)


def Exp():
    return OperationSimple(np.exp, np.exp)


def Sin():
    return OperationSimple(np.sin, np.cos)


def Cos():
    return OperationSimple(np.cos, lambda x: -1*np.sin(x))
