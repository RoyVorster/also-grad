from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union
import numpy as np


# Quick utils
def plural(x):
    return x if isinstance(x, tuple) else (x, )


# Top-level grad enabled flag
enable_grad = True

@contextmanager
def no_grad() -> Iterator[None]:
    enable_grad = False
    try:
        yield
    finally:
        enable_grad = True


# Parameters with gradients
class Parameter:
    def __init__(self, data, requires_grad=True):
        self.data: np.npdarray = np.asarray(data) if not isinstance(data, Parameter) else data.data
        self.grad, self.requires_grad = None, requires_grad

        self.creator = None # An operation

    def detach(self):
        if not self.requires_grad:
            self.creator = None
            return self

        return Parameter(self.data, requires_grad=False)

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def zeros(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def init(cls, *shape: int, **kwargs) -> Parameter:
        # Standard uniform initialization
        data = np.random.uniform(-1, 1, shape)/np.sqrt(np.prod(shape))
        return cls(data.astype(np.float32), **kwargs)

    def backward(self):
        # DFS
        nodes, seen = [], set()
        def dfs_(node):
            # If can go forward, keep going
            if node not in seen and node.creator:
                for new_node in node.creator.parents:
                    dfs_(new_node)

                nodes.append(new_node)

        dfs_(self)

        # Go backward through the graph
        self.grad = Parameter.ones(self.shape, requires_grad=False)
        for node in reversed(nodes):
            if not any(p.requires_grad for parameters in node.creator.parents):
                continue

            grads = node.creator.backward_(n.grad.data)
            for (parent, grad) in zip(node.creator.parents, grads):
                if grad and parent.requires_grad:
                    parent.grad = parent.grad + grad if parent.grad else grad

    # Override with Operations
    def __add__(self, other) -> Parameter:
        return Add()(self, other)

    def __mul__(self, other) -> Parameter:
        return Mul()(self, other)

    def sum(self, axis=None) -> Parameter:
        return Sum()


# Any operation on parameters
class Operation:
    def reset(self):
        self.parents, self.cache = [], []

    def add_to_cache(self, *xs: np.ndarray):
        self.cache += xs

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, g: np.ndarray) -> Union[Parameter, [Parameter]]:
        raise NotImplementedError

    # Internal wrappers
    def forward_(self, *parameters: Parameter) -> np.ndarray:
        self.parents = parameters

        requires_grad = any(p.requires_grad for p in parameters)
        return Parameter(self.forward(*[p.data for p in parameters]), requires_grad=requires_grad)

    def backward_(self, g: np.ndarray) -> [Parameter]:
        return plural(self.backward(g))

    def __call__(self, *parameters) -> Parameter:
        self.reset()

        output = self.forward_(*parameters)
        if enable_grad and output.requires_grad:
            output.creator = self

        return output

