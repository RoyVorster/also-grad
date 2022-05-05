import numpy as np

from alsograd.core import Parameter, Operation


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
