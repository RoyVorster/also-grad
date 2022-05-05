from typing import Tuple, Sequence
import numpy as np

from alsograd.core import Parameter, Operation


# Activations
def relu(a: Parameter) -> Parameter:
    return a.clamp(a_min=0, a_max=None)


def softmax(a: Parameter) -> Parameter:
    new_shape = list(a.shape)
    new_shape[-1] = 1

    err = (a - a.max(axis=-1).reshape(*new_shape)).exp()
    return err/err.sum(axis=-1).reshape(*new_shape)


def log_softmax(a: Parameter) -> Parameter:
    new_shape = list(a.shape)
    new_shape[-1] = 1

    a_max = a.max(axis=-1).reshape(*new_shape)
    return a_max + (a - a_max).exp().sum(axis=-1).reshape(*new_shape).log()


# Losses
def MSE(y_true: Parameter, y_pred: Parameter) -> Parameter:
    return ((y_true - y_pred)**2).mean()


# Functions
def addmm(x: Parameter, b: Parameter, w: Parameter):
    b_shape = [1]*(x.ndim - 1) + [-1]
    return x@w + b.reshape(*b_shape)


class Stack(Operation):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, *aa: np.ndarray) -> np.ndarray:
        assert all(a.shape == aa[0].shape for a in aa), "All shapes should be the same"
        return np.stack(aa, axis=self.axis)

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, ...]:
        return tuple(np.moveaxis(g, self.axis, 0))


def stack(x: Sequence[Parameter], axis: int = -1) -> Parameter:
    return Stack(axis=axis)(x)
