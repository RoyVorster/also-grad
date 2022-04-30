from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from typing import Iterator, Union, List, Tuple, Set, Any, Optional, Callable, Iterable
import numpy as np

from alsograd.utils import plural


# Top-level grad enabled flag
enable_grad = True


@contextmanager
def no_grad() -> Iterator[None]:
    global enable_grad
    enable_grad = False
    try:
        yield
    finally:
        enable_grad = True


# Parameters with gradients
class Parameter:
    def __init__(self, data: Union[Parameter, np.ndarray, Iterable[Any]], requires_grad=True):
        self.data: np.ndarray = data.data if isinstance(data, Parameter) else np.asarray(data)
        self.requires_grad: bool = data.requires_grad if isinstance(data, Parameter) else requires_grad

        self.grad: Optional[Parameter] = None
        self.creator: Optional[Operation] = None

    def __str__(self) -> str:
        s = str(self.data)
        if self.requires_grad:
            if self.grad:
                s += ', with gradient: ' + str(self.grad)
            else:
                s += ', with empty gradient'
        else:
            s += ', with no gradient required'

        return s + '.'

    def detach(self) -> Parameter:
        if not self.requires_grad:
            self.creator = None
            return self

        return Parameter(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad, self.creator = None, None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @classmethod
    def zeros(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def rand(cls, *shape: int, **kwargs) -> Parameter:
        return cls(np.random.rand(*shape).astype(np.float32), **kwargs)

    @classmethod
    def init(cls, *shape: int, **kwargs) -> Parameter:
        # Standard uniform initialization
        data = np.random.uniform(-1, 1, shape)/np.sqrt(np.prod(shape))
        return cls(data.astype(np.float32), **kwargs)

    def backward(self) -> None:
        if not self.requires_grad or not enable_grad:
            return

        # DFS
        nodes: List[Parameter] = []
        seen: Set[Parameter] = set()

        def dfs_(node: Parameter):
            # If can go forward, keep going
            if node not in seen and node.creator:
                for new_node in node.creator.parents:
                    dfs_(new_node)

                nodes.append(node)

        dfs_(self)

        # Go backward through the graph
        self.grad = Parameter(np.ones_like(self.data), requires_grad=False)
        for node in reversed(nodes):
            if not (node and node.creator and node.grad):
                continue

            if not any(p.requires_grad for p in node.creator.parents):
                continue

            grads = node.creator.backward_(node.grad.data)
            for (parent, grad) in zip(node.creator.parents, grads):
                if grad and parent.requires_grad:
                    parent.grad = parent.grad + grad if parent.grad else grad

    # Operations
    def __add__(self, other) -> Parameter:
        return ops.Add()(self, other)

    def __sub__(self, other) -> Parameter:
        return ops.Sub()(self, other)

    def __mul__(self, other) -> Parameter:
        return ops.Mul()(self, other)

    def __div__(self, other) -> Parameter:
        return ops.Div()(self, other)

    def __pow__(self, exp: float) -> Parameter:
        return ops.Pow(exp=exp)(self)

    def __matmul__(self, other) -> Parameter:
        return ops.Dot()(self, other)

    def sum(self, **kwargs) -> Parameter:
        return ops.Sum(**kwargs)(self)


# Any operation on parameters
class Operation:
    forward: Callable[..., np.ndarray]
    backward: Callable[..., Union[np.ndarray, Tuple[np.ndarray, ...]]]

    def reset(self):
        self.parents, self.cache = [], []

    def add_to_cache(self, *xs: Any) -> None:
        self.cache += xs

    # Internal wrappers
    def forward_(self, *parameters: Parameter) -> Parameter:
        self.parents = parameters

        requires_grad = any(p.requires_grad for p in parameters)
        return Parameter(self.forward(*[p.data for p in parameters]), requires_grad=requires_grad)

    def backward_(self, g: np.ndarray) -> List[Parameter]:
        return [Parameter(x, requires_grad=True) for x in plural(self.backward(g))]

    def __call__(self, *parameters) -> Parameter:
        self.reset()

        output = self.forward_(*parameters)
        if enable_grad and output.requires_grad:
            output.creator = copy(self)

        return output


# Circular imports
import alsograd.operations as ops
