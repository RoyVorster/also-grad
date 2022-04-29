import numpy as np

# Backwards pass through sum operation
def rev_sum(x, shape):
    axis = [-1] if shape == (1, ) else [i for i, s in enumerate(shape) if s == 1 and x.shape[i] > 1]
    return x.sum(axis=ax).reshape(shape) if len(ax) > 0 else x

