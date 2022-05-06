from typing import Tuple, Sequence
import numpy as np

from alsograd.core import Parameter, Operation


# Activations
def relu(a: Parameter) -> Parameter:
    return a.clamp(a_min=0, a_max=None)


def sigmoid(a: Parameter) -> Parameter:
    return (1. + (-1.*a).exp())**-1.


def softmax(a: Parameter) -> Parameter:
    new_shape = list(a.shape)[:-1] + [1]

    err = (a - a.max(axis=-1).reshape(*new_shape)).exp()
    return err/err.sum(axis=-1).reshape(*new_shape)


# Losses
def MSE(y_pred: Parameter, y_true: Parameter) -> Parameter:
    return ((y_true - y_pred)**2).mean()


def cross_entropy_loss(y_pred: Parameter, y_true: Parameter) -> Parameter:
    return (-softmax(y_pred).log()*y_true).sum(axis=1).mean()


# Functions
def addmm(x: Parameter, b: Parameter, w: Parameter):
    b_shape = [1]*(x.ndim - 1) + [-1]
    return x@w + b.reshape(*b_shape)


def bmm(x: Parameter, y: Parameter):
    return (x@y).sum(axis=0)


def self_attention(x: Parameter):
    assert x.ndim >= 2
    y = x.transpose(order=(*list(range(x.ndim))[:-2], -1, -2))

    return bmm(x, y)


class Stack(Operation):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, *aa: np.ndarray) -> np.ndarray:
        assert all(a.shape == aa[0].shape for a in aa), "All shapes should be the same"
        return np.stack(aa, axis=self.axis)

    def backward(self, g: np.ndarray) -> Tuple[np.ndarray, ...]:
        return tuple(np.moveaxis(g, self.axis, 0))


def stack(x: Sequence[Parameter], axis: int = -1) -> Parameter:
    return Stack(axis=axis)(*x)
