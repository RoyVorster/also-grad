import numpy as np

from alsograd.core import Parameter, Operation


class ReLU(Operation):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.add_to_cache(a)
        return np.maximum(a, 0)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, = self.cache
        return g*(a >= 0)


def relu(a: Parameter) -> Parameter:
    return ReLU()(a)


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
